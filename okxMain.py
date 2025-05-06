import numpy as np
import pandas as pd
import ta
import ccxt
import time
import urllib3
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 统一配置管理
CONFIG = {
    "SYMBOL": "BTC/USDT",
    "TIMEFRAME": "15m",
    "CHECK_TIMEFRAME": "60m",
    "HISTORY_LIMIT": 200,
    "REFRESH_INTERVAL": 60,
    "PROXY_CONFIG": {
        "http": "socks5://127.0.0.1:7890",
        "https": "socks5://127.0.0.1:7890"
    },
    "INDICATOR_PARAMS": {
        "MACD": (12, 26, 9),
        "RSI_PERIODS": [6],
        "KDJ": {
            "window": 6,
            "smooth_window": 2
        },
        "SMA_PERIODS": [5, 10, 20],
        "BOLLINGER": {
            "window": 20,
            "dev_factor": 2
        },
        "STOCHRSI": {
            "window": 14,
            "smooth1": 3,
            "smooth2": 3
        },
        "ADX_PERIOD": 14,
        "DIVGENCE_WINDOW": 5,
        "ATR_PERIOD": 14
    },
    "TRADING_CONFIG": {
        "OVERBOUGHT_THRESHOLD": {
            "rsi": 70,
            "kdj": 80,
            "srsi": 80
        },
        "OVERSOLD_THRESHOLD": {
            "rsi": 30,
            "kdj": 20,
            "srsi": 20
        },
        "BOLLINGER_BUFFER": {
            "buy": 1.1,
            "sell": 0.9
        },
        "ADX_THRESHOLD": 30,
        "MIN_DATA_LENGTH": 20
    },
    "MODEL_CONFIG": {
        "TEST_SIZE": 0.2,
        "RANDOM_STATE": 42,
        "RF_PARAM_GRID": {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [None, 15, 25, 35],
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 4]
        },
        "MODEL_DIR": "models/",
        "BASE_MODEL_NAME": "trading_model.pkl",
        "MAX_MODELS_TO_KEEP": 5
    }
}


# 早停回调类
class EarlyStoppingCallback:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def __call__(self, model, X_val, y_val):
        current_score = accuracy_score(y_val, model.predict(X_val))
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# 交易所单例模式（带市场数据加载）
class ExchangeSingleton:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            exchange = ccxt.okx({
                "proxies": CONFIG["PROXY_CONFIG"],
                "enableRateLimit": True,
                "timeout": 5000,
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"}
            })
            max_retries = 3
            for i in range(max_retries):
                try:
                    exchange.load_markets()
                    break
                except Exception as e:
                    if i == max_retries - 1:
                        logging.error(f"加载市场数据失败，已达到最大重试次数: {e}")
                        raise
                    logging.warning(f"加载市场数据失败: {e}，重试中...")
                    time.sleep(2)
            cls._instance = exchange
        return cls._instance


exchange = ExchangeSingleton()


# 带重试机制的数据获取
def fetch_ohlcv(symbol, timeframe, limit):
    """获取OHLCV数据并处理异常"""
    for _ in range(3):
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not data:
                logging.warning("获取到空数据，重试中...")
                continue
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            # 先将时间戳转换为无时区的datetime对象
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            # 再将其标记为UTC时区
            df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')
            # 最后移除时区信息
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)
            return df
        except ccxt.ExchangeError as e:
            logging.warning(f"交易所API错误: {str(e)}")
            time.sleep(2)
        except Exception as e:
            logging.error(f"数据获取失败: {e}")
    return None


# 指标计算模块
def compute_kdj(df, window, smooth_window):
    """计算KDJ指标"""
    kdj = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"],
        window=window, smooth_window=smooth_window
    )
    df["kdj_k"] = kdj.stoch()
    df["kdj_d"] = kdj.stoch_signal()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    return df


def compute_sma(df, periods):
    """计算多重移动平均线"""
    for period in periods:
        df[f"sma_{period}"] = ta.trend.SMAIndicator(df["close"], period).sma_indicator()
    df["sma_crossover"] = (df["sma_5"] > df["sma_10"]).astype(int)
    return df


def compute_bollinger_bands(df, window, dev_factor):
    """增加异常价格处理"""
    close = df["close"].where(df["close"] > 0, other=df["close"].shift(1))
    bb = ta.volatility.BollingerBands(close, window, dev_factor)
    df["bollinger_mavg"] = bb.bollinger_mavg()
    df["bollinger_hband"] = bb.bollinger_hband()
    df["bollinger_lband"] = bb.bollinger_lband()
    df["bb_width"] = (df["bollinger_hband"] - df["bollinger_lband"]) / df["bollinger_mavg"]
    return df


def compute_stochrsi_indicator(df, window, smooth1, smooth2):
    """计算Stochastic RSI"""
    srsi = ta.momentum.StochRSIIndicator(df["close"], window, smooth1, smooth2)
    df["srsi"] = srsi.stochrsi_k()
    return df


def compute_adx_indicator(df, period):
    """计算ADX指标"""
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], period)
    df["adx"] = adx.adx()
    return df


# 交易对验证函数（简化版）
def validate_symbol(symbol):
    """检查交易对是否存在"""
    return symbol in exchange.markets


# 增强的成交量过滤
def check_data_validity(df):
    """增强数据有效性检查（包括指标所需最小数据长度）"""
    MIN_INDICATOR_DATA = 50  # 指标计算所需最小数据长度
    if len(df) < MIN_INDICATOR_DATA:
        logging.warning(f"数据长度不足 {MIN_INDICATOR_DATA} 行，跳过")
        return False
    if df["close"].isna().any() or df["volume"].isna().any():
        logging.error("存在收盘价或成交量缺失，数据无效")
        return False
    return True

# 成交量指标计算（带异常处理）
def compute_volume_indicators(df, volume):
    """计算成交量相关指标（带异常处理）"""
    volume = volume.mask(volume <= 0, np.nan)
    df["volume_sma_20"] = ta.trend.SMAIndicator(volume, window=20, fillna=True).sma_indicator()

    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        df["close"], volume, fillna=True
    ).on_balance_volume()

    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["close"], volume=volume
    ).volume_weighted_average_price()

    df["vwap"] = df["vwap"].ffill().fillna(df["bollinger_mavg"])
    return df


def compute_atr_indicator(df, period):
    """计算平均真实波幅"""
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], period)
    df["atr"] = atr.average_true_range()
    return df


def detect_divergence(df, window):
    """检测价格与指标的背离"""
    df["price_high"] = df["high"].rolling(window).max()
    df["macd_high"] = df["macd"].rolling(window).max()
    df["rsi_high"] = df["rsi_6"].rolling(window).max()

    df["bearish_divergence"] = (df["high"] == df["price_high"]) & (
            (df["macd"] < df["macd_high"]) |
            (df["rsi_6"] < df["rsi_high"])
    )

    df["bullish_divergence"] = (df["low"] == df["low"].rolling(window).min()) & (
            (df["macd"] > df["macd"].rolling(window).min()) |
            (df["rsi_6"] > df["rsi_6"].rolling(window).min())
    )
    return df


# def calculate_all_indicators(df):
#     """简化指标计算（保留核心指标）"""
#     close = df["close"]
#     volume = df["volume"]
#
#     df = (df
#     .pipe(compute_kdj, **CONFIG["INDICATOR_PARAMS"]["KDJ"])
#     .pipe(compute_sma, periods=CONFIG["INDICATOR_PARAMS"]["SMA_PERIODS"])
#     .pipe(compute_bollinger_bands, **CONFIG["INDICATOR_PARAMS"]["BOLLINGER"])
#     .assign(
#         macd=ta.trend.MACD(close, *CONFIG["INDICATOR_PARAMS"]["MACD"]).macd(),
#         macd_signal=ta.trend.MACD(close, *CONFIG["INDICATOR_PARAMS"]["MACD"]).macd_signal(),
#         macd_hist=ta.trend.MACD(close, *CONFIG["INDICATOR_PARAMS"]["MACD"]).macd_diff(),
#         rsi_6=ta.momentum.RSIIndicator(close, CONFIG["INDICATOR_PARAMS"]["RSI_PERIODS"][0]).rsi()
#     )
#     .pipe(compute_stochrsi_indicator, **CONFIG["INDICATOR_PARAMS"]["STOCHRSI"])
#     .pipe(compute_adx_indicator, period=CONFIG["INDICATOR_PARAMS"]["ADX_PERIOD"])
#     .pipe(compute_atr_indicator, period=CONFIG["INDICATOR_PARAMS"]["ATR_PERIOD"])
#     .pipe(compute_volume_indicators, volume=volume)
#     .assign(
#         price_deviation=(df["close"] - df["sma_20"]) / df["sma_20"]
#     )
#     )
#
#     return df


# 多周期趋势验证
def validate_multi_timeframe(symbol, main_tf, check_tf, limit):
    """验证主周期与辅助周期趋势一致性"""
    main_df = fetch_ohlcv(symbol, main_tf, limit)
    check_df = fetch_ohlcv(symbol, check_tf, limit)

    if main_df is None or check_df is None:
        return False

    main_df = calculate_all_indicators(main_df)
    check_df = calculate_all_indicators(check_df)

    main_trend = main_df.iloc[-1]["close"] > main_df.iloc[-1]["sma_20"]
    check_trend = check_df.iloc[-1]["close"] > check_df.iloc[-1]["sma_20"]

    return main_trend == check_trend


# 交易信号生成
def generate_trading_signals(df):
    """简化版信号生成（核心条件+辅助增强）"""
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None
    trend = "上涨" if last["close"] > last["sma_20"] else "下跌"

    # 核心条件（必选）
    core_buy_conditions = [
        trend == "上涨",
        last["macd"] > last["macd_signal"],  # MACD金叉
        last["close"] > last["sma_20"]  # 价格在20日均线上
    ]

    # 辅助条件（至少满足2个）
    auxiliary_buy_conditions = [
        last["rsi_6"] < 35,  # 短期超卖
        last["kdj_j"] < 40,
        last["volume"] > last["volume_sma_20"],  # 成交量放大
        validate_multi_timeframe(CONFIG["SYMBOL"], CONFIG["TIMEFRAME"], CONFIG["CHECK_TIMEFRAME"],
                                 CONFIG["HISTORY_LIMIT"])  # 多周期一致
    ]

    buy_signal = len(core_buy_conditions) == 3 and sum(auxiliary_buy_conditions) >= 2

    # 核心条件（必选）
    core_sell_conditions = [
        trend == "下跌",
        last["macd"] < last["macd_signal"],  # MACD死叉
        last["close"] < last["sma_20"]  # 价格在20日均线以下
    ]

    # 辅助条件（至少满足2个）
    auxiliary_sell_conditions = [
        last["rsi_6"] > 65,  # 短期超买
        last["kdj_j"] > 60,
        last["volume"] > last["volume_sma_20"],  # 成交量放大
        validate_multi_timeframe(CONFIG["SYMBOL"], CONFIG["TIMEFRAME"], CONFIG["CHECK_TIMEFRAME"],
                                 CONFIG["HISTORY_LIMIT"])  # 多周期一致
    ]

    sell_signal = len(core_sell_conditions) == 3 and sum(auxiliary_sell_conditions) >= 2

    return {
        "trend": trend,
        "buy": buy_signal,
        "sell": sell_signal,
        "last": last,
        "prev": prev
    }


# 模型加载函数（指定基准模型）
def load_base_model():
    """加载固定名称的基准模型"""
    model_path = Path(CONFIG["MODEL_CONFIG"]["MODEL_DIR"]) / CONFIG["MODEL_CONFIG"]["BASE_MODEL_NAME"]
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            logging.warning(f"基准模型 {CONFIG['MODEL_CONFIG']['BASE_MODEL_NAME']} 加载失败: {e}")
    return None


# 模型训练模块（强制特征对齐）
def train_trading_model(df):
    # 定义严格的特征列表，与指标计算完全一致
    required_features = [
        "rsi_6", "kdj_j", "macd", "macd_hist", "srsi",  # 动量指标
        "volume", "volume_sma_20", "obv", "vwap",  # 成交量指标
        "adx", "atr", "bb_width", "sma_crossover", "price_deviation"  # 趋势与波动指标
    ]

    # 提取特征并删除缺失值
    feature_df = df[required_features].dropna()
    if feature_df.empty:
        logging.error("所有特征均存在缺失值，跳过训练")
        return None

    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    combined = pd.concat([feature_df, df["label"]], axis=1).dropna()  # 合并标签并去空

    if len(combined) < 100:
        logging.info("训练数据不足，跳过")
        return None

    X, y = combined.drop("label", axis=1), combined["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["MODEL_CONFIG"]["TEST_SIZE"],
        random_state=CONFIG["MODEL_CONFIG"]["RANDOM_STATE"]
    )

    # 初始化模型
    model = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=CONFIG["MODEL_CONFIG"]["RANDOM_STATE"],
            n_jobs=-1
        ),
        param_distributions=CONFIG["MODEL_CONFIG"]["RF_PARAM_GRID"],
        n_iter=20,
        cv=5,
        verbose=0
    )
    model.fit(X_train, y_train)
    new_accuracy = model.score(X_test, y_test)

    # 加载基准模型并获取其准确率
    base_model = load_base_model()
    base_accuracy = base_model.score(X_test, y_test) if base_model else 0.5  # 基准模型不存在时以50%为基准

    # 比较准确率并决定是否更新
    model_dir = Path(CONFIG["MODEL_CONFIG"]["MODEL_DIR"])
    base_model_path = model_dir / CONFIG["MODEL_CONFIG"]["BASE_MODEL_NAME"]

    if new_accuracy > base_accuracy:
        # 保存新模型并覆盖基准模型
        joblib.dump(model, base_model_path)
        logging.info(f"模型更新成功！新模型准确率：{new_accuracy:.4f}（原基准：{base_accuracy:.4f}）")
        return model
    else:
        # 不保存新模型，删除可能存在的临时文件（如果之前有保存的话）
        temp_model_path = model_dir / f"trading_model_{time.strftime('%Y%m%d_%H%M%S')}_acc{new_accuracy:.2f}.pkl"
        if temp_model_path.exists():
            temp_model_path.unlink()
        logging.info(f"新模型准确率未超过基准（{new_accuracy:.4f} < {base_accuracy:.4f}），不更新")
        return base_model  # 返回原有基准模型

# 指标计算模块（确保特征生成顺序一致）
def calculate_all_indicators(df):
    close = df["close"]
    volume = df["volume"]

    df = (df
        .pipe(compute_kdj, **CONFIG["INDICATOR_PARAMS"]["KDJ"])
        .pipe(compute_sma, periods=CONFIG["INDICATOR_PARAMS"]["SMA_PERIODS"])
        .pipe(compute_bollinger_bands, **CONFIG["INDICATOR_PARAMS"]["BOLLINGER"])
        .assign(
            macd=ta.trend.MACD(close, *CONFIG["INDICATOR_PARAMS"]["MACD"]).macd(),
            macd_signal=ta.trend.MACD(close, *CONFIG["INDICATOR_PARAMS"]["MACD"]).macd_signal(),
            macd_hist=ta.trend.MACD(close, *CONFIG["INDICATOR_PARAMS"]["MACD"]).macd_diff(),
            rsi_6=ta.momentum.RSIIndicator(close, CONFIG["INDICATOR_PARAMS"]["RSI_PERIODS"][0]).rsi()  # 修正硬编码
        )
        .pipe(compute_stochrsi_indicator, **CONFIG["INDICATOR_PARAMS"]["STOCHRSI"])
        .pipe(compute_adx_indicator, period=CONFIG["INDICATOR_PARAMS"]["ADX_PERIOD"])
        .pipe(compute_atr_indicator, period=CONFIG["INDICATOR_PARAMS"]["ATR_PERIOD"])
        .pipe(compute_volume_indicators, volume=volume)
        .assign(
            sma_crossover=(df["sma_5"] > df["sma_10"]).astype(int),
            price_deviation=(df["close"] - df["sma_20"]) / df["sma_20"]
        )
        .ffill()  # 前向填充缺失值
        .bfill()  # 后向填充剩余缺失值
        .pipe(detect_divergence, window=CONFIG["INDICATOR_PARAMS"]["DIVGENCE_WINDOW"])
    )

    # 强制类型转换（避免数据类型不一致）
    for col in ["volume_sma_20", "adx", "atr", "bb_width"]:
        df[col] = df[col].astype(np.float32)

    return df


def get_model_accuracy(model, X_test, y_test):
    """获取模型准确率"""
    return model.score(X_test, y_test)


# 主执行流程
if __name__ == "__main__":
    # 初始化模型目录
    Path(CONFIG["MODEL_CONFIG"]["MODEL_DIR"]).mkdir(exist_ok=True)

    while True:
        if not validate_symbol(CONFIG["SYMBOL"]):
            logging.error(f"无效交易对: {CONFIG['SYMBOL']}，请检查配置")
            time.sleep(CONFIG["REFRESH_INTERVAL"])
            continue

        ticker_data = fetch_ohlcv(
            symbol=CONFIG["SYMBOL"],
            timeframe=CONFIG["TIMEFRAME"],
            limit=CONFIG["HISTORY_LIMIT"]
        )

        if ticker_data is None or ticker_data.empty:
            logging.warning("数据获取失败，5秒后重试...")
            time.sleep(10)
            continue

        if not check_data_validity(ticker_data):
            time.sleep(10)
            continue

        indicator_df = calculate_all_indicators(ticker_data)
        indicator_df["price_deviation"] = (indicator_df["close"] - indicator_df["sma_20"]) / indicator_df["sma_20"]

        signals = generate_trading_signals(indicator_df)
        last_candle = signals["last"]
        prev_candle = signals["prev"]

        # 打印MACD数值
        try:
            logging.info(
                f"MACD: {last_candle['macd']:.6f}, MACD信号: {last_candle['macd_signal']:.6f}, MACD柱状图: {last_candle['macd_hist']:.6f}")
        except KeyError:
            logging.error("'macd_hist' 列不存在，可能指标计算有误。")

        # 打印金叉死叉情况
        if prev_candle is not None:
            if prev_candle['macd'] <= prev_candle['macd_signal'] and last_candle['macd'] > last_candle['macd_signal']:
                logging.info("【MACD金叉】MACD线向上穿过信号线")
            elif prev_candle['macd'] >= prev_candle['macd_signal'] and last_candle['macd'] < last_candle['macd_signal']:
                logging.info("【MACD死叉】MACD线向下穿过信号线")

        logging.info("\n==================== 实时市场分析 ====================")
        logging.info(f"交易对: {CONFIG['SYMBOL']} | 时间框架: {CONFIG['TIMEFRAME']}")
        logging.info(f"当前价格: {last_candle['close']:.6f} | SMA20: {last_candle['sma_20']:.6f}")
        logging.info(
            f"布林带: {last_candle['bollinger_lband']:.6f} (下轨) - {last_candle['bollinger_mavg']:.6f} (中轨) - {last_candle['bollinger_hband']:.6f} (上轨)")
        try:
            logging.info(
                f"RSI6: {last_candle['rsi_6']:.2f} | KDJJ: {last_candle['kdj_j']:.2f} | SRSI: {last_candle['srsi']:.2f}")
        except KeyError:
            logging.error("'srsi' 列不存在，可能指标计算有误。")
        logging.info(f"成交量: {last_candle['volume']:.2f} | 成交量均线: {last_candle['volume_sma_20']:.2f}")
        try:
            logging.info(f"ADX趋势强度: {last_candle['adx']:.2f} | OBV: {last_candle['obv']:.2f}")
        except KeyError:
            logging.error("'adx' 列不存在，可能指标计算有误。")

        if signals["buy"]:
            logging.info("\n【买入信号】所有买入条件满足，建议开多仓")
            logging.info("关键条件: 上涨趋势 + MACD金叉 + 短期超卖 + 多周期一致 + 底背离")
        elif signals["sell"]:
            logging.info("\n【卖出信号】所有卖出条件满足，建议开空仓")
            logging.info("关键条件: 下跌趋势 + MACD死叉 + 短期超买 + 多周期一致 + 顶背离")
        else:
            logging.info("\n【观望信号】当前无明确交易信号，建议继续监测")
            if last_candle["bullish_divergence"]:
                logging.info("提示: 存在底背离，注意潜在反转机会")
            elif last_candle["bearish_divergence"]:
                logging.info("提示: 存在顶背离，注意潜在回调风险")

        if len(indicator_df) >= 200:
            trained_model = train_trading_model(indicator_df)
            if trained_model:
                logging.info(f"模型训练完成，当前最优模型: {trained_model.best_estimator_}")

        time.sleep(CONFIG["REFRESH_INTERVAL"])
