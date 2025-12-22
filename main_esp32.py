# main_esp32.py
import time
from typing import Optional, Tuple

from esp32_client import ESP32Client
from AI_argo import SleepPreprocessor, WakeDecisionModel, SleepController


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_motion_from_angles(
    prev: Optional[Tuple[float, float, float]],
    cur: Tuple[float, float, float],
    scale: float = 30.0,
) -> Tuple[float, Tuple[float, float, float]]:
    """
    Pitch/Yaw/Roll の変化量から motion_inst を作る（簡易）
    motion_inst = (|ΔPitch|+|ΔYaw|+|ΔRoll|) * scale
    """
    if prev is None:
        return 0.0, cur
    dp = abs(cur[0] - prev[0])
    dy = abs(cur[1] - prev[1])
    dr = abs(cur[2] - prev[2])
    return (dp + dy + dr) * scale, cur


class HRFilter:
    """
    心拍の異常値(-1, レンジ外, 急変)を弾きつつ、短時間なら last_valid を補完する。
    """
    def __init__(self, lo: float = 30.0, hi: float = 220.0, max_jump: float = 50.0, max_stale_sec: float = 5.0):
        self.lo = lo
        self.hi = hi
        self.max_jump = max_jump
        self.max_stale_sec = max_stale_sec
        self.last_valid: Optional[float] = None
        self.last_valid_ts: Optional[float] = None

    def update(self, t_sec: float, hr_raw: float) -> Tuple[Optional[float], str]:
        # -1 / 0 は無効
        if hr_raw <= 0:
            return self._impute_or_none(t_sec, hr_raw, reason="hr<=0")

        # レンジ外
        if not (self.lo <= hr_raw <= self.hi):
            return self._impute_or_none(t_sec, hr_raw, reason="range_out")

        # 急変（センサ誤検出が多い）
        if self.last_valid is not None and abs(hr_raw - self.last_valid) > self.max_jump:
            return self._impute_or_none(t_sec, hr_raw, reason="jump_out")

        # OK
        self.last_valid = hr_raw
        self.last_valid_ts = t_sec
        return hr_raw, "ok"

    def _impute_or_none(self, t_sec: float, hr_raw: float, reason: str) -> Tuple[Optional[float], str]:
        if self.last_valid is None or self.last_valid_ts is None:
            return None, f"invalid({reason})"

        if (t_sec - self.last_valid_ts) <= self.max_stale_sec:
            # 短時間は last_valid を補完
            return self.last_valid, f"imputed({reason})"
        return None, f"invalid({reason})"


class StableDecisionPolicy:
    """
    probの揺れでPWMがチカチカしないための安定化：
    - EMAでprob平滑化
    - ヒステリシス（ONとOFFの閾値を分ける）
    - ラッチ（ONになったら一定時間はONを維持）
    """
    def __init__(self, on_th: float = 0.55, off_th: float = 0.45, ema_alpha: float = 0.25, latch_sec: float = 10.0):
        self.on_th = on_th
        self.off_th = off_th
        self.ema_alpha = ema_alpha
        self.latch_sec = latch_sec

        self.prob_ema: Optional[float] = None
        self.state_on: bool = False
        self.latched_until: float = -1.0

    def step(self, t_sec: float, prob: float, forced: bool) -> Tuple[int, float, bool]:
        # EMA
        if self.prob_ema is None:
            self.prob_ema = prob
        else:
            self.prob_ema = self.ema_alpha * prob + (1.0 - self.ema_alpha) * self.prob_ema

        # forcedなら強制ON＋ラッチ延長
        if forced:
            self.state_on = True
            self.latched_until = max(self.latched_until, t_sec + self.latch_sec)
            return 1, float(self.prob_ema), True

        # ラッチ中はON維持
        if t_sec < self.latched_until:
            self.state_on = True
            return 1, float(self.prob_ema), False

        # ヒステリシス判定
        if not self.state_on and self.prob_ema >= self.on_th:
            self.state_on = True
            self.latched_until = t_sec + self.latch_sec
        elif self.state_on and self.prob_ema <= self.off_th:
            self.state_on = False

        return (1 if self.state_on else 0), float(self.prob_ema), False


def main():
    client = ESP32Client(base_url="http://192.168.4.1", timeout=1.0)

    print("=== ESP32 status ===")
    print(client.get_status())

    # ===== テスト用（短い許容時間）=====
    # 本番はアプリ設定で allow_start_ts/allow_end_ts を決める
    sleep_start_ts = 0.0
    allow_start_ts = 1 * 60.0
    allow_end_ts   = 2 * 60.0

    # まずはデバッグ設定（あなたのログ確認がしやすい）
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

    #hr_filter = HRFilter(lo=30.0, hi=220.0, max_jump=60.0, max_stale_sec=5.0)
    hr_filter = HRFilter(lo=30.0, hi=220.0, max_jump=120.0, max_stale_sec=10.0)

    policy = StableDecisionPolicy(on_th=0.55, off_th=0.45, ema_alpha=0.25, latch_sec=10.0)

    print("=== ループ開始 ===")

    t0_ms: Optional[int] = None
    prev_angles: Optional[Tuple[float, float, float]] = None

    # 起動時消灯
    try:
        client.set_pwm(0.0)
    except Exception:
        pass

    while True:
        loop_start = time.time()

        # 1) sensors
        try:
            s = client.get_sensors()
        except Exception as e:
            print("[WARN] get_sensors failed:", e)
            # 通信失敗は安全側
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

        # 2) forcedゾーン判定（HRが死んでも絶対起こすため）
        time_to_end_min = max(0.0, (allow_end_ts - t_sec) / 60.0)
        forced_zone = (time_to_end_min <= model.force_wake_margin_min)

        # 3) HRフィルタ（forced中でも「補完」できるならする）
        hr_used, hr_status = hr_filter.update(t_sec, hr_raw)

        print(f"[RAW] t={t_sec:6.2f}s hr_raw={hr_raw:7.1f} hr_used={str(hr_used):>7}({hr_status}) "
              f"motion={motion_inst:6.2f} time_to_end={time_to_end_min:5.2f}min forced_zone={forced_zone}")

        # 4) forcedゾーン中は、HRが無効でも“起こす”を優先
        if forced_zone:
            pwm = 30.0  # forced時はもう少し強くしても良い（好みで 60 など）
            pwm = clamp(pwm, 0.0, 100.0)
            try:
                client.set_pwm(pwm)
            except Exception as e:
                print("[WARN] set_pwm failed:", e)
            print(f"  -> FORCED OUTPUT pwm={pwm:4.1f}")
            time.sleep(max(0.0, 1.0 - (time.time() - loop_start)))
            continue

        # 5) forcedでない通常時：HRが完全に取れないなら「今回は判定しない」
        if hr_used is None:
            # 無効が続くと窓が薄くなるので、ここは「消灯」か「維持」か方針が必要。
            # ここでは安全側：消灯
            try:
                client.set_pwm(0.0)
            except Exception:
                pass
            print("  -> HRが取れないためAI投入スキップ（消灯）")
            time.sleep(max(0.0, 1.0 - (time.time() - loop_start)))
            continue

        # 6) AI投入
        result = controller.on_sample(timestamp=t_sec, hr_inst=float(hr_used), motion_inst=motion_inst)
        if result is None:
            print("  -> no-decision（warmup or interval待ち）")
        else:
            decision_raw, prob_raw, debug, feats = result
            forced = bool(debug.get("forced", False))

            # 安定化
            decision, prob_ema, latched_by_forced = policy.step(t_sec, prob_raw, forced)

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
