# main_esp32.py
import time
from typing import Optional, Tuple

from esp32_client import ESP32Client
from AI_argo import SleepPreprocessor, WakeDecisionModel, SleepController


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_motion_from_angles(prev: Optional[Tuple[float, float, float]],
                               cur: Tuple[float, float, float]) -> Tuple[float, Tuple[float, float, float]]:
    if prev is None:
        return 0.0, cur
    dp = abs(cur[0] - prev[0])
    dy = abs(cur[1] - prev[1])
    dr = abs(cur[2] - prev[2])
    return (dp + dy + dr), cur


def main():
    client = ESP32Client(base_url="http://192.168.4.1", timeout=1.0)

    print("=== ESP32 status ===")
    print(client.get_status())

    # ===== テストを早くするため短め設定（本番では window_sec=60 に戻す）=====
    sleep_start_ts = 0.0
    allow_start_ts = 1 * 60.0   # 1分後
    allow_end_ts   = 2 * 60.0   # 2分後

    preproc = SleepPreprocessor(
        sleep_start_ts=sleep_start_ts,
        allow_start_ts=allow_start_ts,
        allow_end_ts=allow_end_ts,
        window_sec=10.0,             # ★まず10秒でテスト（本番は60秒推奨）
        decision_interval_sec=1.0,   # ★毎秒判定（本番は10秒でもOK）
        motion_still_threshold=5.0,
        hr_base_fixed=None,
        baseline_minutes=1.0,        # ★短縮（本番は5分など）
    )
    model = WakeDecisionModel()
    controller = SleepController(preproc, model)

    print("=== ループ開始 ===")
    t0_ms: Optional[int] = None
    prev_angles: Optional[Tuple[float, float, float]] = None

    # 安全：起動時消灯
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
            time.sleep(0.2)
            continue

        # 2) 必須キーをログ（毎回出す）
        ts_ms = int(s.get("timestamp", 0))
        hr = float(s.get("heart_rate", -1))
        pitch = float(s.get("Pitch", 0.0))
        yaw = float(s.get("Yaw", 0.0))
        roll = float(s.get("Roll", 0.0))

        if t0_ms is None:
            t0_ms = ts_ms
        t_sec = (ts_ms - t0_ms) / 1000.0

        motion_inst, prev_angles = compute_motion_from_angles(prev_angles, (pitch, yaw, roll))

        print(f"[RAW] t={t_sec:6.2f}s ts_ms={ts_ms} hr={hr:6.1f} "
              f"Pitch/Yaw/Roll=({pitch:6.2f},{yaw:6.2f},{roll:6.2f}) motion={motion_inst:6.2f}")

        # 3) 心拍無効なら理由を出してスキップ（ここが今まで無言だった）
        if hr <= 0:
            print("  -> heart_rate が無効（<=0 または -1）。センサ装着/初期化を確認。")
            time.sleep(0.5)
            continue

        # 4) AI投入
        result = controller.on_sample(timestamp=t_sec, hr_inst=hr, motion_inst=motion_inst)

        if result is None:
            print("  -> warmup（窓がまだ埋まってないので判定なし）")
        else:
            decision, prob, debug, feats = result
            pwm = 20.0 if decision == 1 else 0.0   # ★安全のため弱め
            pwm = clamp(pwm, 0.0, 100.0)

            try:
                client.set_pwm(pwm)
            except Exception as e:
                print("[WARN] set_pwm failed:", e)

            print(
                f"  -> AI dec={decision} prob={prob:5.3f} pwm={pwm:4.1f} "
                f"(hr_mean={feats.hr_mean:5.1f} hrv_sd={feats.hrv_sd:4.1f} "
                f"motion_sum={feats.motion_sum:6.1f} still={feats.still_minutes:4.2f}min "
                f"time_to_end={feats.time_to_end:5.2f}min forced={debug.get('forced', False)})"
            )

        # 5) 1Hzに近づける
        elapsed = time.time() - loop_start
        time.sleep(max(0.0, 1.0 - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n停止しました（Ctrl+C）")
