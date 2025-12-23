# main_esp32.py
import time
from typing import Optional, Tuple

from esp32_client import ESP32Client
from AI_argo import SleepPreprocessor, WakeDecisionModel, SleepController

import kalman_filter as kfmod  # ← pybind11でビルドしたモジュール


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_motion_from_angles(
    prev: Optional[Tuple[float, float, float]],
    cur: Tuple[float, float, float],
    scale: float = 30.0,
) -> Tuple[float, Tuple[float, float, float]]:
    if prev is None:
        return 0.0, cur
    dp = abs(cur[0] - prev[0])
    dy = abs(cur[1] - prev[1])
    dr = abs(cur[2] - prev[2])
    return (dp + dy + dr) * scale, cur


class KalmanHR:
    """
    C++ KalmanFilter を使って心拍を平滑化する。
    - 有効値: predict + update
    - 欠損/異常: predictのみ（updateしない）
    - 欠損が長い場合は None を返してAI投入を止める
    """
    def __init__(
        self,
        Q: float = 0.5,
        R: float = 4.0,
        P0: float = 10.0,
        init_fallback: float = 70.0,
        hr_lo: float = 30.0,
        hr_hi: float = 220.0,
        max_stale_sec: float = 10.0,
    ):
        self.Q = Q
        self.R = R
        self.P0 = P0
        self.init_fallback = init_fallback
        self.hr_lo = hr_lo
        self.hr_hi = hr_hi
        self.max_stale_sec = max_stale_sec

        self._kf: Optional[kfmod.KalmanFilter] = None
        self._last_update_t: Optional[float] = None

    def _ensure_init(self, z: float, t_sec: float):
        x0 = z if z > 0 else self.init_fallback
        self._kf = kfmod.KalmanFilter(float(x0), float(self.P0), float(self.Q), float(self.R))
        self._last_update_t = t_sec

    def step(self, t_sec: float, hr_raw: float) -> Tuple[Optional[float], str]:
        # 未初期化なら、最初の有効値が来るまで待つ
        if self._kf is None:
            if self.hr_lo <= hr_raw <= self.hr_hi:
                self._ensure_init(hr_raw, t_sec)
                return float(self._kf.getState()), "init"
            return None, "not_init"

        # まず予測（欠損でもやる）
        self._kf.predict()

        # 観測が有効か？
        if not (self.hr_lo <= hr_raw <= self.hr_hi):
            # 欠損/異常：updateしない
            if self._last_update_t is None or (t_sec - self._last_update_t) > self.max_stale_sec:
                return None, "missing(stale)"
            return float(self._kf.getState()), "missing(hold)"

        # 観測更新
        x = float(self._kf.update(float(hr_raw)))
        self._last_update_t = t_sec
        return x, "ok"


class StableDecisionPolicy:
    """
    チカチカ防止：EMA + ヒステリシス + ラッチ
    """
    def __init__(self, on_th: float = 0.55, off_th: float = 0.45, ema_alpha: float = 0.25, latch_sec: float = 10.0):
        self.on_th = on_th
        self.off_th = off_th
        self.ema_alpha = ema_alpha
        self.latch_sec = latch_sec

        self.prob_ema: Optional[float] = None
        self.state_on: bool = False
        self.latched_until: float = -1.0

    def step(self, t_sec: float, prob: float, forced: bool) -> Tuple[int, float]:
        # EMA
        if self.prob_ema is None:
            self.prob_ema = prob
        else:
            self.prob_ema = self.ema_alpha * prob + (1.0 - self.ema_alpha) * self.prob_ema

        # forcedは強制ON（+ラッチ延長）
        if forced:
            self.state_on = True
            self.latched_until = max(self.latched_until, t_sec + self.latch_sec)
            return 1, float(self.prob_ema)

        # ラッチ中はON維持
        if t_sec < self.latched_until:
            self.state_on = True
            return 1, float(self.prob_ema)

        # ヒステリシス
        if (not self.state_on) and (self.prob_ema >= self.on_th):
            self.state_on = True
            self.latched_until = t_sec + self.latch_sec
        elif self.state_on and (self.prob_ema <= self.off_th):
            self.state_on = False

        return (1 if self.state_on else 0), float(self.prob_ema)


def main():
    client = ESP32Client(base_url="http://192.168.4.1", timeout=1.0)

    print("=== ESP32 status ===")
    print(client.get_status())
    print("=== ループ開始 ===")

    # ===== テスト用（短い許容時間）=====
    sleep_start_ts = 0.0
    allow_start_ts = 1 * 60.0
    allow_end_ts   = 2 * 60.0

    preproc = SleepPreprocessor(
        sleep_start_ts=sleep_start_ts,
        allow_start_ts=allow_start_ts,
        allow_end_ts=allow_end_ts,
        window_sec=10.0,
        decision_interval_sec=1.0,
        motion_still_threshold=5.0,
        hr_base_fixed=None,
        baseline_minutes=1.0,
    )
    model = WakeDecisionModel()
    controller = SleepController(preproc, model)

    # ★ここが差し替えポイント：HRFilter → KalmanHR
    hr_kalman = KalmanHR(Q=0.5, R=4.0, P0=10.0, max_stale_sec=10.0)

    policy = StableDecisionPolicy(on_th=0.55, off_th=0.45, ema_alpha=0.25, latch_sec=10.0)

    t0_ms: Optional[int] = None
    prev_angles: Optional[Tuple[float, float, float]] = None

    # 起動時消灯
    try:
        client.set_pwm(0.0)
    except Exception:
        pass

    while True:
        loop_start = time.time()

        # 1) 取得
        try:
            s = client.get_sensors()
        except Exception as e:
            print("[WARN] get_sensors failed:", e)
            try:
                client.set_pwm(0.0)
            except Exception:
                pass
            time.sleep(0.2)
            continue

        ts_ms = int(s.get("timestamp", 0))
        if t0_ms is None:
            t0_ms = ts_ms
        t_sec = (ts_ms - t0_ms) / 1000.0

        hr_raw = float(s.get("heart_rate", -1))
        pitch = float(s.get("Pitch", 0.0))
        yaw   = float(s.get("Yaw", 0.0))
        roll  = float(s.get("Roll", 0.0))

        motion_inst, prev_angles = compute_motion_from_angles(prev_angles, (pitch, yaw, roll), scale=30.0)

        # 2) forcedゾーン（ここは“絶対起こす”を最優先）
        time_to_end_min = max(0.0, (allow_end_ts - t_sec) / 60.0)
        forced_zone = (time_to_end_min <= model.force_wake_margin_min)

        # 3) カルマンで心拍を平滑化
        hr_used, hr_status = hr_kalman.step(t_sec, hr_raw)

        print(f"[RAW] t={t_sec:6.2f}s hr_raw={hr_raw:7.1f} hr_used={str(hr_used):>7}({hr_status}) "
              f"motion={motion_inst:6.2f} time_to_end={time_to_end_min:5.2f}min forced_zone={forced_zone}")

        # 4) forced中はAIを通さずPWM確定（HR欠損でも保証を守る）
        if forced_zone:
            pwm = 30.0
            try:
                client.set_pwm(clamp(pwm, 0.0, 100.0))
            except Exception as e:
                print("[WARN] set_pwm failed:", e)
            print(f"  -> FORCED OUTPUT pwm={pwm:4.1f}")
            time.sleep(max(0.0, 1.0 - (time.time() - loop_start)))
            continue

        # 5) forcedでない通常時：心拍が全く作れないならAI投入しない
        if hr_used is None:
            try:
                client.set_pwm(0.0)
            except Exception:
                pass
            print("  -> HRが作れないためAI投入スキップ（消灯）")
            time.sleep(max(0.0, 1.0 - (time.time() - loop_start)))
            continue

        # 6) AI投入
        result = controller.on_sample(timestamp=t_sec, hr_inst=float(hr_used), motion_inst=motion_inst)
        if result is None:
            print("  -> no-decision（warmup or interval待ち）")
        else:
            decision_raw, prob_raw, debug, feats = result
            forced = bool(debug.get("forced", False))

            decision, prob_ema = policy.step(t_sec, prob_raw, forced)

            pwm = 20.0 if decision == 1 else 0.0
            try:
                client.set_pwm(pwm)
            except Exception as e:
                print("[WARN] set_pwm failed:", e)

            print(
                f"  -> AI raw(dec={decision_raw} prob={prob_raw:5.3f}) "
                f"stable(dec={decision} prob_ema={prob_ema:5.3f}) "
                f"pwm={pwm:4.1f} feats(hr_mean={feats.hr_mean:5.1f} hrv_sd={feats.hrv_sd:4.1f} "
                f"motion_sum={feats.motion_sum:6.1f} still={feats.still_minutes:4.2f}min)"
            )

        # 7) 1Hz
        elapsed = time.time() - loop_start
        time.sleep(max(0.0, 1.0 - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n停止しました（Ctrl+C）")
