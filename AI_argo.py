from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple, Dict, List
import math


# ========================
# 1. 特徴量のデータ構造
# ========================

@dataclass
class SleepFeatures:
    """
    前処理済みの特徴量セット（全部 float 前提）
    """
    hr_base: float          # 安静時心拍の推定値
    hr_mean: float          # 直近1分の平均心拍
    hrv_sd: float           # 直近1分の心拍揺れ（標準偏差など）

    motion: float           # 予備（今は未使用だが保持）
    motion_sum: float       # 1分間の体動量
    delta_motion: float     # 前の1分との体動変化

    still_minutes: float    # "ほぼ動いてない"状態が続いている分数

    t: float                # 現在の「寝始めからの経過時間」（分）
    t_start: float          # 起床許容開始時刻（寝始めからの分）
    t_end: float            # 起床許容終了時刻（寝始めからの分）
    sleep_elapsed: float    # = t と同じ（意味を分かりやすくするために別名で持つ）
    time_to_end: float      # 起床許容終了までの残り時間（分）

    delta_hr: float         # 前の1分との平均心拍変化


# ========================
# 2. 評価モデル本体
# ========================

class WakeDecisionModel:
    """
    光目覚まし用の評価アルゴリズム本体。
    1分ごとに特徴量から「起こす/待つ（1/0）」を決定する。
    """

    def __init__(self):
        # 正規化用のレンジ
        self.hr_rel_min = -0.30   # 安静時より30%低い
        self.hr_rel_max =  0.30   # 安静時より30%高い

        self.hrv_min = 0.0
        self.hrv_max = 80.0

        self.motion_min = 0.0
        self.motion_max = 500.0

        self.still_min = 0.0
        self.still_max = 30.0     # 30分以上静止で最大扱い

        self.sleep_elapsed_min = 0.0
        self.sleep_elapsed_max = 480.0  # 8時間

        self.delta_hr_max_abs = 20.0       # bpm
        self.delta_motion_max_abs = 300.0  # 体動変化

        # 評価関数の重み
        self.weights: Dict[str, float] = {
            "f_hr_rel":         0.4,   # 基礎心拍より高いほど浅い
            "f_hrv":            0.3,   # HRV大きいほど浅い/レム寄り
            "f_motion":         0.9,   # 今よく動いているほど起こしやすい
            "f_still":         -0.8,   # 長時間静止は深い睡眠→マイナス
            "f_sleep_progress": 0.3,   # 夜の後半ほど起こしやすい
            "f_time_pressure":  1.2,   # 許容終了に近いほど強く起こす
            "f_delta_hr":       0.2,   # 心拍上昇中なら起こしやすい
            "f_delta_motion":   0.5,   # 動き増加中なら起こしやすい
        }
        self.bias = -0.6  # 全体バイアス
        self.threshold = 0.5

        # 許容終了何分前で「必ず1」にするか（デフォルト: 0.5分 = 30秒）
        # time_to_end が分単位で渡ってくる前提。
        self.force_wake_margin_min = 0.5

    # --- ユーティリティ ---

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _norm01(self, x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        v = (x - lo) / (hi - lo)
        return self._clamp(v, 0.0, 1.0)

    def _norm_signed(self, x: float, max_abs: float) -> float:
        if max_abs <= 0:
            return 0.0
        v = x / max_abs
        return self._clamp(v, -1.0, 1.0)

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x < -60.0:
            return 0.0
        if x > 60.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    # --- 正規化＋特徴量生成 ---

    def _normalize_features(self, f: SleepFeatures) -> Dict[str, float]:
        # 心拍の相対変化
        if f.hr_base > 0:
            hr_rel = (f.hr_mean - f.hr_base) / f.hr_base
        else:
            hr_rel = 0.0

        f_hr_rel = self._norm_signed(
            hr_rel,
            max_abs=max(abs(self.hr_rel_min), abs(self.hr_rel_max))
        )

        f_hrv = self._norm01(f.hrv_sd, self.hrv_min, self.hrv_max)
        f_motion = self._norm01(f.motion_sum, self.motion_min, self.motion_max)
        f_still = self._norm01(f.still_minutes, self.still_min, self.still_max)

        f_sleep_progress = self._norm01(
            f.sleep_elapsed, self.sleep_elapsed_min, self.sleep_elapsed_max
        )

        # 許容時間内の進行度（0〜1）
        if f.t_end > f.t_start:
            time_ratio = (f.t - f.t_start) / (f.t_end - f.t_start)
        else:
            time_ratio = 0.0
        f_time_pressure = self._clamp(time_ratio, 0.0, 1.0)

        f_delta_hr = self._norm_signed(f.delta_hr, self.delta_hr_max_abs)
        f_delta_motion = self._norm_signed(
            f.delta_motion, self.delta_motion_max_abs
        )

        return {
            "f_hr_rel":         f_hr_rel,
            "f_hrv":            f_hrv,
            "f_motion":         f_motion,
            "f_still":          f_still,
            "f_sleep_progress": f_sleep_progress,
            "f_time_pressure":  f_time_pressure,
            "f_delta_hr":       f_delta_hr,
            "f_delta_motion":   f_delta_motion,
        }

    # --- メイン評価 ---

    def evaluate(self, f: SleepFeatures) -> Tuple[int, float, Dict[str, float]]:
        """
        1ステップ分の判定を行う。

        Returns:
            decision: 0 = 「待つ」, 1 = 「照射してよい」
            prob:     スコア（0〜1, 起こしてよさそう度）
            debug:    正規化後の特徴量と z を含むデバッグ情報
        """

        # ★ 強制起床ゾーン（許容終了30秒前など）★
        if f.time_to_end <= self.force_wake_margin_min:
            debug = {
                "forced": True,
                "reason": "time_margin",
                "time_to_end": f.time_to_end,
                "z_raw": float("inf"),
                "prob": 1.0,
                "threshold": self.threshold,
            }
            return 1, 1.0, debug

        # 通常評価
        nf = self._normalize_features(f)

        z = self.bias
        for name, value in nf.items():
            w = self.weights.get(name, 0.0)
            z += w * value

        prob = self._sigmoid(z)
        decision = 1 if prob >= self.threshold else 0

        debug = dict(nf)
        debug["z_raw"] = z
        debug["prob"] = prob
        debug["threshold"] = self.threshold
        debug["forced"] = False
        debug["reason"] = "normal_decision"

        return decision, prob, debug


# ========================
# 3. 前処理クラス
# ========================

class SleepPreprocessor:
    """
    生データ (timestamp, hr_inst, motion_inst) を受け取り、
    window_sec の窓で SleepFeatures を生成するクラス。
    （判定間隔は decision_interval_sec）
    """

    def __init__(
        self,
        sleep_start_ts: float,
        allow_start_ts: float,
        allow_end_ts: float,
        window_sec: float = 60.0,
        decision_interval_sec: float = 60.0,
        motion_still_threshold: float = 5.0,
        hr_base_fixed: Optional[float] = None,
        baseline_minutes: float = 5.0,
    ):
        """
        Args:
            sleep_start_ts:   睡眠開始時刻（秒）
            allow_start_ts:   起床許容開始時刻（秒）
            allow_end_ts:     起床許容終了時刻（秒）
            window_sec:       特徴量計算に使う時間窓（秒）
            decision_interval_sec: 判定を出す間隔（秒）
            motion_still_threshold: これ未満なら「静止」とみなす体動量
            hr_base_fixed:    安静時心拍の事前推定値。Noneなら自動推定
            baseline_minutes: 自動推定に使う時間（分）
        """
        self.sleep_start_ts = sleep_start_ts
        self.allow_start_ts = allow_start_ts
        self.allow_end_ts = allow_end_ts

        self.window_sec = window_sec
        self.decision_interval_sec = decision_interval_sec

        self.motion_still_threshold = motion_still_threshold

        # サンプルバッファ（直近 window_sec 分）
        self.samples: deque[Tuple[float, float, float]] = deque()
        # 要素: (timestamp_sec, hr_inst, motion_inst)

        self.prev_timestamp: Optional[float] = None

        # 静止時間の積算（秒）
        self.still_duration_sec: float = 0.0

        # 直前の窓の平均値（差分計算用）
        self.prev_hr_mean: Optional[float] = None
        self.prev_motion_sum: Optional[float] = None

        # hr_base
        self.hr_base: Optional[float] = hr_base_fixed
        self.baseline_needed_sec = baseline_minutes * 60.0
        self.baseline_hr_accum: float = 0.0
        self.baseline_hr_time: float = 0.0  # ベースラインの積算（ここではサンプル数扱い）

        # 最後に判定を出した時刻
        self.last_decision_ts: Optional[float] = None

        # t, t_start, t_end を「寝始めからの分」であらかじめ計算
        self.t_start_min = (self.allow_start_ts - self.sleep_start_ts) / 60.0
        self.t_end_min = (self.allow_end_ts - self.sleep_start_ts) / 60.0

    # ---- サンプル追加 & 静止時間計算 ----

    def add_sample(self, timestamp: float, hr_inst: float, motion_inst: float
                   ) -> Optional[SleepFeatures]:
        """
        生データ1サンプルを追加。
        必要に応じて新しい SleepFeatures を返す（それ以外は None）。

        ★修正点：
        実機ではサンプル間隔が微妙に揺れるため、
        初回判定の「窓が埋まった判定」を (window_sec - 平均サンプル間隔) まで緩める。
        これにより warmup が永遠に終わらない現象を防ぐ。
        """
        # サンプル追加
        self.samples.append((timestamp, hr_inst, motion_inst))

        # 古すぎるサンプルを捨てる
        window_start = timestamp - self.window_sec
        while self.samples and self.samples[0][0] < window_start:
            self.samples.popleft()

        # 静止時間を更新
        self._update_still_duration(timestamp, motion_inst)

        # hr_base 自動推定
        self._update_hr_base(timestamp, hr_inst)

        # 判定を出すタイミングか？
        if self.last_decision_ts is None:
            # ---- 初回判定 ----
            if not self.samples:
                return None

            coverage = self.samples[-1][0] - self.samples[0][0]  # 窓の実効幅
            n = len(self.samples)

            # 平均サンプル間隔を推定（n<2なら0）
            dt_est = coverage / max(1, n - 1)

            # 1サンプル分だけ“不足”を許す（実機の揺らぎで届かないのを防ぐ）
            need = max(0.0, self.window_sec - dt_est)

            if coverage >= need:
                self.last_decision_ts = timestamp
                return self._compute_features(timestamp)
            else:
                return None

        else:
            # ---- 通常判定 ----
            if timestamp - self.last_decision_ts >= self.decision_interval_sec:
                self.last_decision_ts = timestamp
                return self._compute_features(timestamp)
            else:
                return None

    def _update_still_duration(self, timestamp: float, motion_inst: float):
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp
            return

        dt = timestamp - self.prev_timestamp
        self.prev_timestamp = timestamp

        if motion_inst < self.motion_still_threshold:
            self.still_duration_sec += dt
        else:
            self.still_duration_sec = 0.0

    def _update_hr_base(self, timestamp: float, hr_inst: float):
        """
        hr_base が未指定のとき、
        最初の baseline_minutes 分から安静時心拍を推定する。
        """
        if self.hr_base is not None:
            return

        # ベースライン計測は sleep_start_ts から baseline_needed_sec まで
        if timestamp <= self.sleep_start_ts + self.baseline_needed_sec:
            # ここではサンプル数で平均（簡易）
            self.baseline_hr_accum += hr_inst
            self.baseline_hr_time += 1.0
        else:
            # まだ決まっていなければ決める
            if self.baseline_hr_time > 0:
                self.hr_base = self.baseline_hr_accum / self.baseline_hr_time
            else:
                self.hr_base = 60.0

    # ---- 特徴量計算 ----

    def _compute_features(self, timestamp: float) -> Optional[SleepFeatures]:
        if not self.samples:
            return None

        # サンプルをリスト化
        hrs: List[float] = []
        motions: List[float] = []

        for _, h, m in self.samples:
            hrs.append(h)
            motions.append(m)

        n = len(hrs)
        if n == 0:
            return None

        hr_mean = sum(hrs) / n

        # 標準偏差
        if n > 1:
            var = sum((h - hr_mean) ** 2 for h in hrs) / (n - 1)
            hrv_sd = math.sqrt(var)
        else:
            hrv_sd = 0.0

        motion_sum = sum(abs(m) for m in motions)
        motion_mean = sum(motions) / n

        # delta系の計算
        if self.prev_hr_mean is None:
            delta_hr = 0.0
        else:
            delta_hr = hr_mean - self.prev_hr_mean

        if self.prev_motion_sum is None:
            delta_motion = 0.0
        else:
            delta_motion = motion_sum - self.prev_motion_sum

        self.prev_hr_mean = hr_mean
        self.prev_motion_sum = motion_sum

        # 時間系の特徴量（分）
        t_min = (timestamp - self.sleep_start_ts) / 60.0
        sleep_elapsed = t_min
        time_to_end = max(0.0, (self.allow_end_ts - timestamp) / 60.0)

        still_minutes = self.still_duration_sec / 60.0

        # hr_base がまだ決まっていない場合の fallback
        hr_base = self.hr_base if self.hr_base is not None else hr_mean

        return SleepFeatures(
            hr_base=hr_base,
            hr_mean=hr_mean,
            hrv_sd=hrv_sd,
            motion=motion_mean,
            motion_sum=motion_sum,
            delta_motion=delta_motion,
            still_minutes=still_minutes,
            t=t_min,
            t_start=self.t_start_min,
            t_end=self.t_end_min,
            sleep_elapsed=sleep_elapsed,
            time_to_end=time_to_end,
            delta_hr=delta_hr,
        )


# ========================
# 4. コントローラクラス（前処理→評価をつなげる）
# ========================

class SleepController:
    """
    生データ→前処理→評価モデル までを一発でつなげるラッパー。
    """

    def __init__(self, preproc: SleepPreprocessor, model: WakeDecisionModel):
        self.preproc = preproc
        self.model = model

    def on_sample(self, timestamp: float, hr_inst: float, motion_inst: float
                  ) -> Optional[Tuple[int, float, Dict[str, float], SleepFeatures]]:
        """
        生サンプルを1つ受け取り、
        必要に応じて decision を返す。

        戻り値:
            None  -> まだ窓が埋まっていないなどで判定なし
            (decision, prob, debug, features)
        """
        features = self.preproc.add_sample(timestamp, hr_inst, motion_inst)
        if features is None:
            return None

        decision, prob, debug = self.model.evaluate(features)
        return decision, prob, debug, features
