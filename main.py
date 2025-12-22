# main.py
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import time
from dataclasses import asdict
from typing import Optional, Tuple

from esp32_client import ESP32Client
from AI_argo import SleepPreprocessor, WakeDecisionModel, SleepController


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class MotionEstimator:
    """
    /api/sensors の Pitch/Yaw/Roll の差分から “体動っぽい量” を作る。
    目的：SleepPreprocessor に入れる motion_inst を安定に生成する。

    - 角度差の絶対値和（deg）を使う
    - スケールを掛けて “閾値5.0” 周辺で反応する程度に調整可能
    """
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._prev: Optional[Tuple[float, float, float]] = None

    def update(self, pitch: Optional[float], yaw: Optional[float], roll: Optional[float]) -> float:
        if pitch is None or yaw is None or roll is None:
            return 0.0

        cur = (float(pitch), float(yaw), float(roll))
        if self._prev is None:
            self._prev = cur
            return 0.0

        dp = abs(cur[0] - self._prev[0])
        dy = abs(cur[1] - self._prev[1])
        dr = abs(cur[2] - self._prev[2])
        self._prev = cur

        # “動いた感” = 角度差の和（deg） * scale
        return (dp + dy + dr) * self.scale


class PWMController:
    """
    PWM値をいきなり変えずに、最大変化量を制限して “チカチカ” を防ぐ。
    """
    def __init__(self, max_step_per_sec: float = 15.0):
        self.max_step_per_sec = max_step_per_sec
        self.current = 0.0

    def step_to(self, target: float) -> float:
        target = clamp(target, 0.0, 100.0)
        diff = target - self.current
        step = clamp(diff, -self.max_step_per_sec, self.max_step_per_sec)
        self.current += step
        self.current = clamp(self.current, 0.0, 100.0)
        return self.current


def prob_to_pwm(prob: float, threshold: float, pwm_min: float, pwm_max: float) -> float:
    """
    prob を PWM にマッピング（なめらか版）
    - prob < threshold なら 0
    - threshold 以上なら (pwm_min〜pwm_max) に線形で上げる
    """
    if prob < threshold:
        return 0.0
    x = (prob - threshold) / (1.0 - threshold + 1e-9)
    x = clamp(x, 0.0, 1.0)
    return pwm_min + x * (pwm_max - pwm_min)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCON 光目覚まし: ESP32 Web API + AI loop")
    p.add_argument("--base-url", default="http://192.168.4.1")
    p.add_argument("--poll-hz", type=float, default=1.0, help="ESP32からの取得周波数（推奨1Hz）")

    # SleepPreprocessor settings
    p.add_argument("--window-sec", type=float, default=60.0)
    p.add_argument("--decision-interval-sec", type=float, default=10.0)
    p.add_argument("--motion-still-threshold", type=float, default=5.0)

    # 起床許容時間（sleep_start=0基準の“相対分”で指定）
    p.add_argument("--allow-start-min", type=float, default=360.0, help="許容開始（分）例: 6h=360")
    p.add_argument("--allow-end-min", type=float, default=420.0, help="許容終了（分）例: 7h=420")

    # PWM behavior
    p.add_argument("--pwm-min", type=float, default=30.0, help="起こす時の最低PWM")
    p.add_argument("--pwm-max", type=float, default=90.0, help="起こす時の最大PWM")
    p.add_argument("--pwm-step", type=float, default=15.0, help="1秒あたりPWM最大変化量")
    p.add_argument("--dry-run", action="store_true", help="PWMを送らずログだけ流す")

    # motion scale
    p.add_argument("--motion-scale", type=float, default=1.0, help="Pitch/Yaw/Roll差分のスケール")

    # logging
    p.add_argument("--csv", default="run_log.csv", help="ログCSV出力パス（空なら出さない）")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    client = ESP32Client(base_url=args.base_url, timeout=1.0, retries=3, backoff_sec=0.2)

    print("=== ESP32 status ===")
    try:
        print(client.get_status())
    except Exception as e:
        print("ESP32に接続できません:", e)
        print("Wi-Fiが ESP32-C3_AP に繋がっているか確認してください。")
        return

    # 時間基準：
    # ESP32の timestamp(ms) を “t=0起点” に揃えて SleepPreprocessor に入れる
    t0_ms: Optional[float] = None

    # SleepPreprocessor の time軸も「秒」で入れる設計なので
    sleep_start_ts = 0.0
    allow_start_ts = args.allow_start_min * 60.0
    allow_end_ts = args.allow_end_min * 60.0

    preproc = SleepPreprocessor(
        sleep_start_ts=sleep_start_ts,
        allow_start_ts=allow_start_ts,
        allow_end_ts=allow_end_ts,
        window_sec=args.window_sec,
        decision_interval_sec=args.decision_interval_sec,
        motion_still_threshold=args.motion_still_threshold,
        hr_base_fixed=None,          # 実機なら自動推定でOK
        baseline_minutes=5.0,
    )
    model = WakeDecisionModel()
    controller = SleepController(preproc, model)

    motion_est = MotionEstimator(scale=args.motion_scale)
    pwm_ctrl = PWMController(max_step_per_sec=args.pwm_step)

    # CSV logger
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "pc_time",
            "esp_timestamp_ms",
            "t_sec",
            "heart_rate",
            "pitch", "yaw", "roll",
            "motion_inst",
            "decision", "prob", "pwm_target", "pwm_sent",
            # 参考（特徴量の一部）
            "hr_mean", "hrv_sd", "motion_sum", "still_minutes", "time_to_end_min",
        ])
        csv_file.flush()

    period = 1.0 / max(0.1, args.poll_hz)
    print("=== loop start (Ctrl+C to stop) ===")

    try:
        while True:
            loop_start = time.time()
            pc_time = dt.datetime.now().isoformat(timespec="seconds")

            # --- fetch sensors ---
            try:
                s = client.get_sensors()
            except Exception as e:
                # フェイルセーフ：通信失敗時はPWMを落とす
                if not args.dry_run:
                    try:
                        client.set_pwm(0.0)
                    except Exception:
                        pass
                print(f"[{pc_time}] sensors fetch failed: {e}")
                time.sleep(period)
                continue

            esp_ts_ms = float(s.get("timestamp", -1))
            heart_rate = float(s.get("heart_rate", -1))

            pitch = s.get("Pitch", None)
            yaw = s.get("Yaw", None)
            roll = s.get("Roll", None)

            # timestamp基準化
            if esp_ts_ms >= 0 and t0_ms is None:
                t0_ms = esp_ts_ms

            if t0_ms is None or esp_ts_ms < 0:
                # タイムスタンプが無いならスキップ（ログは残す）
                print(f"[{pc_time}] invalid timestamp: {s}")
                time.sleep(period)
                continue

            t_sec = (esp_ts_ms - t0_ms) / 1000.0
            if t_sec < 0:
                # まれにタイムスタンプ巻き戻りがあったら基準を張り直す
                t0_ms = esp_ts_ms
                t_sec = 0.0

            # 心拍無効は -1 が仕様
            if heart_rate <= 0:
                # 無効なら「学習窓を汚さない」方針で、サンプル投入しない
                print(f"[{pc_time}] hr invalid (heart_rate={heart_rate}). skip.")
                time.sleep(period)
                continue

            # motion推定（Pitch/Yaw/Roll差分）
            motion_inst = motion_est.update(pitch, yaw, roll)

            # --- AI step ---
            result = controller.on_sample(timestamp=t_sec, hr_inst=heart_rate, motion_inst=motion_inst)

            decision = ""
            prob = ""
            pwm_target = 0.0
            pwm_sent = 0.0

            hr_mean = ""
            hrv_sd = ""
            motion_sum = ""
            still_minutes = ""
            time_to_end_min = ""

            if result is None:
                # ウォームアップ中
                print(f"[{pc_time}] warmup t={t_sec:7.1f}s hr={heart_rate:5.1f} motion={motion_inst:6.2f}")
            else:
                dec_i, prob_f, debug, feats = result
                decision = int(dec_i)
                prob = float(prob_f)

                # PWMは prob でなめらかに（＋ランプ制限）
                pwm_target = prob_to_pwm(prob_f, model.threshold, args.pwm_min, args.pwm_max)
                pwm_sent = pwm_ctrl.step_to(pwm_target)

                if not args.dry_run:
                    try:
                        client.set_pwm(pwm_sent)
                    except Exception as e:
                        print(f"[{pc_time}] set_pwm failed: {e}")

                # 表示（最低限）
                print(
                    f"[{pc_time}] t={t_sec:7.1f}s hr={heart_rate:5.1f} "
                    f"prob={prob_f:5.3f} dec={dec_i} pwm={pwm_sent:5.1f} "
                    f"motion={motion_inst:6.2f}"
                )

                # ログ用
                hr_mean = feats.hr_mean
                hrv_sd = feats.hrv_sd
                motion_sum = feats.motion_sum
                still_minutes = feats.still_minutes
                time_to_end_min = feats.time_to_end

            # CSV
            if csv_writer is not None:
                csv_writer.writerow([
                    pc_time,
                    esp_ts_ms,
                    f"{t_sec:.3f}",
                    f"{heart_rate:.3f}",
                    pitch, yaw, roll,
                    f"{motion_inst:.6f}",
                    decision, prob,
                    f"{pwm_target:.3f}", f"{pwm_sent:.3f}",
                    hr_mean, hrv_sd, motion_sum, still_minutes, time_to_end_min,
                ])
                csv_file.flush()

            # --- pacing ---
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, period - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping... (set PWM=0)")
        if not args.dry_run:
            try:
                client.set_pwm(0.0)
            except Exception:
                pass
    finally:
        client.close()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
