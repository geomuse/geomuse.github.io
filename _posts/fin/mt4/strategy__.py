import talib
import numpy as np
import joblib
from xgboost import XGBClassifier

class MartingaleStrategy:
    def __init__(self):
        # ===== 加载模型（支持 walk-forward 模型池） =====
        try:
            self.models = joblib.load("xgb_walkforward_models.pkl")
            if isinstance(self.models, list) and len(self.models) == 0:
                print("[WARN] Model list is empty")
                self.models = None
                self.model_ready = False
            else:
                self.model_ready = True
        except Exception:
            print("[WARN] Using empty model (for testing only)")
            self.models = None
            self.model_ready = False

        # ===== 技术参数 =====
        self.ema_trend = 200
        self.rsi_period = 14
        self.atr_period = 14
        self.pip = 0.01

        # ===== 交易参数 =====
        self.initial_lots = 0.01
        self.martingale_multiplier = 1.4
        self.max_martingale_orders = 5
        self.max_pyramid_orders = 3

        # ===== 风控 =====
        self.take_profit_usd = 6.0
        self.risk_pct = 0.02
        self.trailing_start = 3.0
        self.trailing_step = 1.0

        # ===== ML 阈值 =====
        self.buy_threshold = 0.60
        self.sell_threshold = 0.60

        self.reset()
        self.initial_balance = None

    def reset(self):
        self.current_direction = None
        self.martingale_count = 0
        self.pyramid_count = 0
        self.last_add_price = 0.0
        self.max_run_profit = 0.0

    # ==========================
    # Regime 判定（趋势/震荡）✅
    # ==========================
    def detect_regime(self, closes, atr):
        if len(closes) < self.ema_trend:
            return "RANGE"
        ema200 = talib.EMA(closes, self.ema_trend)[-1]
        price = closes[-1]
        dist = abs(price - ema200) / atr if atr > 0 else 0
        return "TREND" if dist > 1.5 else "RANGE"

    # ==========================
    # 特征工程
    # ==========================
    def build_features(self, data):
        closes = np.array(data["closes"], dtype=float)
        highs = np.array(data["highs"], dtype=float)
        lows = np.array(data["lows"], dtype=float)

        # ✅ 确保数据足够
        if len(closes) < 10:  # 最小10个 bar
            return None, None, None

        rsi_series = talib.RSI(closes, self.rsi_period)
        ema_series = talib.EMA(closes, self.ema_trend)
        atr_series = talib.ATR(highs, lows, closes, self.atr_period)

        rsi = rsi_series[-1]
        ema200 = ema_series[-1]
        atr = atr_series[-1]

        if np.isnan(rsi) or np.isnan(ema200) or np.isnan(atr):
            return None, None, None

        # ✅ 特征计算，保证索引安全
        ret = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        volatility = np.std(closes[-10:]) if len(closes) >= 10 else np.std(closes)
        ema_dist = (closes[-1] - ema200) / ema200

        features = np.array([[rsi, atr, ret, volatility, ema_dist]])
        regime = self.detect_regime(closes, atr)
        return features, atr, regime


    # ==========================
    # 主逻辑
    # ==========================
    def on_tick(self, data):
        bid, ask = float(data["bid"]), float(data["ask"])
        orders = int(data["orders"])
        profit = float(data["profit"])
        balance = float(data["balance"])
        bar_index = int(data.get("bar_index", 0))  # ✅ walk-forward 模型索引

        if self.initial_balance is None:
            self.initial_balance = balance

        # ===== 风控 =====
        if profit > self.max_run_profit:
            self.max_run_profit = profit

        if self.max_run_profit >= self.trailing_start and profit <= self.max_run_profit - self.trailing_step:
            self.reset()
            return {"action": "CLOSE_ALL", "msg": "Trailing Stop"}

        if profit <= -balance * self.risk_pct:
            self.reset()
            return {"action": "CLOSE_ALL", "msg": "Risk Stop"}

        if profit >= self.take_profit_usd:
            self.reset()
            return {"action": "CLOSE_ALL", "msg": "Hard TP"}

        # ===== 构建特征 & regime ✅
        features, atr, regime = self.build_features(data)
        if features is None or not self.model_ready:
            return None

        # ===== Walk-Forward 模型选择 ✅
        if isinstance(self.models, list):
            model_index = min(len(self.models) - 1, bar_index // 500)
            model = self.models[model_index]
        else:
            model = self.models

        prob_up = model.predict_proba(features)[0][1]

        # ===== 过滤低置信度区（强烈建议）✅
        if 0.45 < prob_up < 0.55:
            return None

        # ===== 初始进场 =====
        if orders == 0:
            if prob_up >= self.buy_threshold:
                self.current_direction = "BUY"
                self.last_add_price = ask
                return {"action": "BUY", "lots": self.initial_lots, "msg": f"ML BUY ({prob_up:.2f})"}
            if prob_up <= 1 - self.sell_threshold:
                self.current_direction = "SELL"
                self.last_add_price = bid
                return {"action": "SELL", "lots": self.initial_lots, "msg": f"ML SELL ({1-prob_up:.2f})"}
            return None

        # ===== ATR 步长 =====
        step_pips = atr / self.pip
        curr_price = bid if self.current_direction == "BUY" else ask

        # ===== 顺势金字塔 =====
        favorable = (curr_price - self.last_add_price) / self.pip \
            if self.current_direction == "BUY" else \
            (self.last_add_price - curr_price) / self.pip

        # ✅ Regime Filter 控制金字塔和马丁
        max_pyramid = self.max_pyramid_orders if regime == "TREND" else 1
        max_martingale = self.max_martingale_orders if regime == "TREND" else 0

        if favorable >= step_pips * 2 and self.pyramid_count < max_pyramid:
            if (self.current_direction == "BUY" and prob_up > 0.55) or \
               (self.current_direction == "SELL" and prob_up < 0.45):
                self.pyramid_count += 1
                self.last_add_price = curr_price
                return {"action": self.current_direction, "lots": self.initial_lots, "msg": "ML Pyramid"}

        # ===== 逆势轻马丁 =====
        unfavorable = (self.last_add_price - curr_price) / self.pip \
            if self.current_direction == "BUY" else \
            (curr_price - self.last_add_price) / self.pip

        if unfavorable >= step_pips and self.martingale_count < max_martingale:
            if (self.current_direction == "BUY" and prob_up > 0.50) or \
               (self.current_direction == "SELL" and prob_up < 0.50):
                self.martingale_count += 1
                self.last_add_price = curr_price
                lots = round(self.initial_lots * (self.martingale_multiplier ** self.martingale_count), 2)
                return {"action": self.current_direction, "lots": lots, "msg": "ML Martingale"}

        return None
