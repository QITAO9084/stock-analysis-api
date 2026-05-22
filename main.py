from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import json
import threading

app = FastAPI(
    title="Stock Analysis API",
    description="股票/加密货币分析API - V5（含买卖点检测、缓存重试限速）",
    version="5.2.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== yfinance 缓存 + 重试 + 限速机制 =====
_yf_cache = {}       # {key: {"data": DataFrame, "info": dict, "ts": float}}
_yf_cache_lock = threading.Lock()
_CACHE_TTL = 3600    # 缓存有效期 1 小时（秒）
_MAX_RETRIES = 5     # 限流时最大重试次数
_RETRY_BASE_DELAY = 4  # 重试基础等待秒数
_last_request_ts = 0.0  # 上次 yfinance 请求时间戳
_request_lock = threading.Lock()
_MIN_REQUEST_INTERVAL = 3.0  # 两次 yfinance 请求最小间隔（秒）
_STALE_CACHE_TTL = 7200  # 过期缓存在限流时可用的最大年龄（秒）


def _cache_key(symbol: str) -> str:
    """生成缓存 key"""
    return symbol.upper()


def _is_rate_limit_error(exc: Exception) -> bool:
    """判断是否为 yfinance 限流错误"""
    msg = str(exc).lower()
    return any(kw in msg for kw in ["rate limit", "too many requests", "429", "timed out"])


def _rate_limit_wait():
    """请求限速：确保两次 yfinance 请求之间至少间隔 _MIN_REQUEST_INTERVAL 秒"""
    global _last_request_ts
    with _request_lock:
        now = time.time()
        elapsed = now - _last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
        _last_request_ts = time.time()


def fetch_yf_data(symbol: str, period: str = "6mo"):
    """
    带 缓存 + 重试 + 限速 + stale-while-revalidate 的 yfinance 数据获取

    返回: (ticker_info, history_dataframe)
    如果全部重试失败且有过期缓存，返回过期缓存（降级）
    如果既无缓存也无数据，抛出最后的异常
    """
    key = _cache_key(symbol)
    now = time.time()

    # 1. 先查缓存（有效缓存直接返回）
    with _yf_cache_lock:
        if key in _yf_cache:
            cached = _yf_cache[key]
            if now - cached["ts"] < _CACHE_TTL:
                return cached["info"], cached["data"].copy()

    # 2. 限速等待 + 请求 + 重试
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            _rate_limit_wait()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            data = ticker.history(period=period)

            if data.empty:
                # 数据为空但没报错，检查是否有过期缓存兜底
                with _yf_cache_lock:
                    if key in _yf_cache:
                        return _yf_cache[key]["info"], _yf_cache[key]["data"].copy()
                return info, data

            # 写入缓存
            with _yf_cache_lock:
                _yf_cache[key] = {
                    "info": info,
                    "data": data.copy(),
                    "ts": now,
                }
            return info, data

        except Exception as e:
            last_exc = e
            if _is_rate_limit_error(e) and attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)  # 4s, 8s, 16s, 32s
                time.sleep(delay)
            else:
                # 最后一次重试也失败了，检查过期缓存兜底
                with _yf_cache_lock:
                    if key in _yf_cache:
                        stale = _yf_cache[key]
                        if now - stale["ts"] < _STALE_CACHE_TTL:
                            # 返回过期缓存作为降级数据
                            return stale["info"], stale["data"].copy()
                raise

    # 所有重试耗尽，最后查一次过期缓存
    with _yf_cache_lock:
        if key in _yf_cache:
            stale = _yf_cache[key]
            if now - stale["ts"] < _STALE_CACHE_TTL:
                return stale["info"], stale["data"].copy()

    raise last_exc

def calculate_rsi(data, period=14):
    """计算RSI指标（返回完整序列，用于趋势判断）"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    prev = rsi.iloc[-2] if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else current
    return current, prev, round(current - prev, 2)

def calculate_macd(data):
    """计算MACD指标（含交叉检测）"""
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    # 交叉检测：前一日 vs 当日
    macd_now = macd.iloc[-1]
    sig_now = signal.iloc[-1]
    macd_prev = macd.iloc[-2] if len(macd) > 1 else macd_now
    sig_prev = signal.iloc[-2] if len(signal) > 1 else sig_now

    # 金叉：前一交易日DIF<DEA，今日DIF>=DEA
    golden_cross = (macd_prev < sig_prev) and (macd_now >= sig_now)
    # 死叉：前一交易日DIF>DEA，今日DIF<=DEA
    death_cross = (macd_prev > sig_prev) and (macd_now <= sig_now)

    # 柱状图趋势：柱子在放大还是缩小
    hist_now = histogram.iloc[-1]
    hist_prev = histogram.iloc[-2] if len(histogram) > 1 else hist_now

    return {
        "macd": round(macd_now, 4),
        "signal": round(sig_now, 4),
        "histogram": round(hist_now, 4),
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "histogram_trend": "expanding" if abs(hist_now) > abs(hist_prev) else "shrinking"
    }

def calculate_kdj(data, period=9):
    """计算KDJ指标"""
    low_list = data['Low'].rolling(window=period).min()
    high_list = data['High'].rolling(window=period).max()
    rsv = (data['Close'] - low_list) / (high_list - low_list) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return {
        "k": round(k.iloc[-1], 2),
        "d": round(d.iloc[-1], 2),
        "j": round(j.iloc[-1], 2)
    }

def calculate_bollinger_bands(data, period=20):
    """计算布林带"""
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return {
        "upper": round(upper.iloc[-1], 2),
        "middle": round(sma.iloc[-1], 2),
        "lower": round(lower.iloc[-1], 2)
    }

def calculate_volume_signal(data):
    """成交量分析：对比近20日均量"""
    if len(data) < 20:
        return "normal", 1.0
    avg_volume = data['Volume'].iloc[-20:-1].mean()
    current_volume = data['Volume'].iloc[-1]
    ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    if ratio > 2.0:
        return "high_volume", round(ratio, 2)
    elif ratio > 1.5:
        return "above_avg", round(ratio, 2)
    elif ratio < 0.5:
        return "low_volume", round(ratio, 2)
    else:
        return "normal", round(ratio, 2)

def get_trading_signal(data, symbol):
    """
    生成交易信号（V2增强版）
    新增：MACD交叉检测、KDJ评分、RSI趋势、成交量确认、MA50大趋势
    """
    current_price = data['Close'].iloc[-1]
    ma5 = data['Close'].rolling(window=5).mean().iloc[-1]
    ma10 = data['Close'].rolling(window=10).mean().iloc[-1]
    ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
    # 新增MA50
    ma50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else None

    rsi, rsi_prev, rsi_delta = calculate_rsi(data)
    macd_data = calculate_macd(data)
    kdj_data = calculate_kdj(data)
    boll = calculate_bollinger_bands(data)
    vol_status, vol_ratio = calculate_volume_signal(data)

    buy_score = 0
    sell_score = 0
    buy_signals = []
    sell_signals = []

    # ---- 第一阶段：独立指标评分 ----

    # 1. RSI评分（含趋势判断）
    if rsi < 30:
        buy_signals.append(f"RSI超卖（{round(rsi,1)}），可能反弹")
        buy_score += 2
    elif rsi < 45:
        buy_signals.append(f"RSI偏低（{round(rsi,1)}），可考虑建仓")
        buy_score += 1

    # RSI趋势加分：超买但正在回落（比单纯超买更温和）
    if rsi > 70 and rsi_delta < -2:
        sell_signals.append(f"RSI超买且快速下降（{round(rsi,1)}→变化{rsi_delta}），注意回调")
        sell_score += 2
    elif rsi > 70:
        sell_signals.append(f"RSI超买（{round(rsi,1)}），注意风险")
        sell_score += 1
    elif rsi > 60:
        sell_signals.append(f"RSI偏高（{round(rsi,1)}），可考虑减仓")
        sell_score += 1

    # RSI从超卖回升 = 强买入信号
    if rsi_prev < 30 and rsi >= 30:
        buy_signals.append(f"RSI从超卖区回升（{round(rsi_prev,1)}→{round(rsi,1)}），反弹信号")
        buy_score += 3

    # 2. MACD评分（含交叉检测）
    if macd_data['golden_cross']:
        buy_signals.append("MACD今日金叉，强烈买入信号")
        buy_score += 3
    elif macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
        buy_signals.append("MACD多头运行，趋势偏多")
        buy_score += 2

    if macd_data['death_cross']:
        sell_signals.append("MACD今日死叉，强烈卖出信号")
        sell_score += 3
    elif macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
        sell_signals.append("MACD空头运行，趋势偏空")
        sell_score += 2

    # MACD柱状图趋势：柱子缩小 = 动能衰减
    if macd_data['histogram_trend'] == "shrinking" and abs(macd_data['histogram']) > 0.5:
        if macd_data['histogram'] > 0:
            sell_signals.append("MACD多头柱缩小，上涨动能衰减")
            sell_score += 1
        else:
            buy_signals.append("MACD空头柱缩小，下跌动能衰减")
            buy_score += 1

    # 3. KDJ评分（新增！）
    k, d, j = kdj_data['k'], kdj_data['d'], kdj_data['j']
    # KDJ金叉（K从下穿越D）
    if len(data) >= 2:
        prev_k_series = (data['Close'] - data['Low'].rolling(9).min()) / (data['High'].rolling(9).max() - data['Low'].rolling(9).min()) * 100
        prev_k = prev_k_series.ewm(com=2, adjust=False).mean().iloc[-2] if len(prev_k_series) > 1 and not pd.isna(prev_k_series.iloc[-2]) else k
        prev_d_val = k  # 简化处理
    else:
        prev_k = k

    if k < 20 and d < 20:
        buy_signals.append(f"KDJ超卖区（K={k}，D={d}），可能反弹")
        buy_score += 2
    elif k > 80 and d > 80:
        sell_signals.append(f"KDJ超买区（K={k}，D={d}），注意风险")
        sell_score += 2
    # J值极端（J > 100 或 J < 0）
    if j > 100:
        sell_signals.append(f"KDJ的J值严重超买（{round(j,1)}），短期回调风险大")
        sell_score += 2
    elif j < 0:
        buy_signals.append(f"KDJ的J值严重超卖（{round(j,1)}），短期反弹可能大")
        buy_score += 2

    # 4. 布林带（不变）
    if current_price < boll['lower']:
        buy_signals.append("价格触及布林带下轨，超卖")
        buy_score += 1
    if current_price > boll['upper']:
        sell_signals.append("价格触及布林带上轨，超买")
        sell_score += 1

    # 5. 均线系统（加入MA50大趋势判断）
    if current_price > ma5 and ma5 > ma10 and ma10 > ma20:
        buy_signals.append("短期均线多头排列，趋势向上")
        buy_score += 2

    if current_price < ma5 and ma5 < ma10 and ma10 < ma20:
        sell_signals.append("短期均线空头排列，趋势向下")
        sell_score += 2

    # MA50大趋势判断
    if ma50 is not None:
        if current_price > ma50 and ma20 > ma50:
            buy_signals.append(f"价格站上50日均线（MA50={round(ma50,2)}），中长期趋势偏多")
            buy_score += 2
        elif current_price < ma50 and ma20 < ma50:
            sell_signals.append(f"价格跌破50日均线（MA50={round(ma50,2)}），中长期趋势偏空")
            sell_score += 2

    # 6. 成交量确认（新增！）
    if vol_status == "high_volume":
        # 放量 + 上涨 = 多头确认
        if data['Close'].iloc[-1] > data['Open'].iloc[-1]:
            buy_signals.append(f"放量上涨（量比{vol_ratio}倍），多头确认")
            buy_score += 2
        else:
            # 放量 + 下跌 = 空头恐慌
            sell_signals.append(f"放量下跌（量比{vol_ratio}倍），空头恐慌")
            sell_score += 2
    elif vol_status == "low_volume":
        # 缩量 = 观望
        buy_signals.append(f"成交量萎缩（量比{vol_ratio}倍），市场观望情绪浓")
        # 缩量不加分不扣分，仅提示

    # ---- 第二阶段：否决机制 ----
    # RSI极端超买（>85）→ 强制否决买入信号
    if rsi > 85 and buy_score > 0:
        sell_signals.insert(0, f"RSI极度超买（{round(rsi,1)}），强烈建议减仓或观望")
        sell_score += 4
    elif rsi > 80 and buy_score > 0:
        sell_signals.insert(0, f"RSI严重超买（{round(rsi,1)}），建议减仓")
        sell_score += 3

    # RSI极端超卖（<15）→ 强制否决卖出信号
    if rsi < 15 and sell_score > 0:
        buy_signals.insert(0, f"RSI极度超卖（{round(rsi,1)}），强烈建议建仓")
        buy_score += 4
    elif rsi < 20 and sell_score > 0:
        buy_signals.insert(0, f"RSI严重超卖（{round(rsi,1)}），建议建仓")
        buy_score += 3

    # ---- 第三阶段：综合评分，决定最终信号 ----
    if buy_score > sell_score and buy_score >= 3:
        signal_type = "BUY"
        signals = buy_signals
        confidence = "HIGH" if buy_score >= 7 else "MEDIUM"
    elif sell_score > buy_score and sell_score >= 3:
        signal_type = "SELL"
        signals = sell_signals
        confidence = "HIGH" if sell_score >= 7 else "MEDIUM"
    else:
        signal_type = "HOLD"
        confidence = "LOW"
        signals = []
        if len(buy_signals) == 0 and len(sell_signals) == 0:
            signals.append("无明显买卖信号，建议观望")
        else:
            signals = buy_signals + sell_signals

    # 计算支撑位和阻力位
    recent_high = data['High'].tail(20).max()
    recent_low = data['Low'].tail(20).min()

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "signal": signal_type,
        "confidence": confidence,
        "signals": signals,
        "indicators": {
            "rsi": round(rsi, 2),
            "rsi_prev": round(rsi_prev, 2),
            "rsi_delta": rsi_delta,
            "macd": macd_data,
            "kdj": kdj_data,
            "bollinger_bands": boll,
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2) if ma50 else None,
        },
        "volume_signal": vol_status,
        "volume_ratio": vol_ratio,
        "support_level": round(recent_low, 2),
        "resistance_level": round(recent_high, 2),
        "trend_direction": "bullish" if (buy_score > sell_score) else ("bearish" if (sell_score > buy_score) else "neutral"),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def read_root():
    """API健康检查"""
    return {
        "status": "ok",
        "message": "Stock Analysis API is running",
        "version": "1.0.0"
    }

@app.get("/stock/info")
def get_stock_info(symbol: str = "AAPL", market: str = "us"):
    """
    获取股票基本信息
    
    - **symbol**: 股票代码（如 AAPL, 00700）
    - **market**: 市场（us/hk/cn）
    """
    # 处理Coze可能传入的"auto"参数
    if symbol == "auto" or not symbol:
        symbol = "AAPL"
    if market == "auto" or not market:
        market = "us"

    symbol, market = normalize_stock_symbol(symbol, market)

    try:
        info, _ = fetch_yf_data(symbol, period="1d")

        return {
            "symbol": symbol,
            "name": info.get("longName", "N/A"),
            "current_price": info.get("currentPrice", 0),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
            "volume": info.get("volume", 0),
            "currency": info.get("currency", "USD"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")

@app.get("/stock/kline")
def get_kline_data(symbol: str = "AAPL", market: str = "us", period: str = "1mo"):
    """
    获取K线数据
    
    - **symbol**: 股票代码
    - **market**: 市场（us/hk/cn）
    - **period**: 时间周期（1d/5d/1mo/3mo/6mo/1y）
    """
    # 处理Coze可能传入的"auto"参数
    if symbol == "auto" or not symbol:
        symbol = "AAPL"
    if market == "auto" or not market:
        market = "us"

    symbol, market = normalize_stock_symbol(symbol, market)

    try:
        _, data = fetch_yf_data(symbol, period=period)

        if data.empty:
            raise HTTPException(status_code=404, detail="未找到股票数据")
        
        kline = []
        for index, row in data.iterrows():
            kline.append({
                "date": index.strftime("%Y-%m-%d"),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })
        
        return {
            "symbol": symbol,
            "period": period,
            "data": kline
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")

@app.get("/stock/signal")
def get_trading_signal_api(symbol: str = "AAPL", market: str = "us"):
    """
    获取买卖信号
    
    - **symbol**: 股票代码
    - **market**: 市场（us/hk/cn）
    """
    # 处理Coze可能传入的"auto"参数
    if symbol == "auto" or not symbol:
        symbol = "AAPL"
    if market == "auto" or not market:
        market = "us"

    symbol, market = normalize_stock_symbol(symbol, market)

    try:
        _, data = fetch_yf_data(symbol, period="3mo")
        
        if data.empty:
            raise HTTPException(status_code=404, detail="未找到股票数据")
        
        return get_trading_signal(data, symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成交易信号失败: {str(e)}")

@app.get("/stock/analyze")
def analyze_stock(symbol: str = "AAPL", market: str = "us"):
    """
    完整股票分析（Coze AI Agent 主要调用接口）

    一次返回：股票信息 + 买卖信号 + K线数据
    Coze Agent 只需调用这一个接口即可完成全部分析

    - **symbol**: 股票代码（如 AAPL, 00700.HK）
    - **market**: 市场（us/hk/cn）
    """
    # 处理Coze可能传入的"auto"参数
    if symbol == "auto" or not symbol:
        symbol = "AAPL"
    if market == "auto" or not market:
        market = "us"

    symbol, market = normalize_stock_symbol(symbol, market)

    try:
        # 获取股票信息 + 历史数据（带缓存和重试）
        info, data = fetch_yf_data(symbol, period="6mo")

        if data.empty:
            raise HTTPException(status_code=404, detail="未找到股票数据")

        # 计算技术指标和信号
        signal_data = get_trading_signal(data, symbol)

        # 获取近期K线（最近30个交易日）
        recent_kline = []
        for index, row in data.tail(30).iterrows():
            recent_kline.append({
                "date": index.strftime("%Y-%m-%d"),
                "open": round(row['Open'], 2),
                "high": round(row['High'], 2),
                "low": round(row['Low'], 2),
                "close": round(row['Close'], 2),
                "volume": int(row['Volume'])
            })

        current_price = round(data['Close'].iloc[-1], 2)
        prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else current_price
        change_percent = round((current_price - prev_close) / prev_close * 100, 2) if prev_close != 0 else 0

        return {
            "symbol": symbol,
            "name": info.get("longName", "N/A"),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": info.get("currency", "USD"),
            "market": market,
            "analysis_time": datetime.now().isoformat(),
            "signal": signal_data["signal"],
            "confidence": signal_data["confidence"],
            "key_signals": signal_data["signals"],
            "rsi": signal_data["indicators"]["rsi"],
            "macd": signal_data["indicators"]["macd"],
            "kdj": signal_data["indicators"]["kdj"],
            "bollinger_bands": signal_data["indicators"]["bollinger_bands"],
            "ma5": signal_data["indicators"]["ma5"],
            "ma10": signal_data["indicators"]["ma10"],
            "ma20": signal_data["indicators"]["ma20"],
            "kline_data": recent_kline,
            "stock_info": {
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "52w_high": info.get("fiftyTwoWeekHigh", 0),
                "52w_low": info.get("fiftyTwoWeekLow", 0),
                "volume": info.get("volume", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析股票失败: {str(e)}")

@app.get("/stock/analyze2")
def analyze_stock_flat(symbol: str = "AAPL", market: str = "us"):
    """
    扁平化股票分析接口（专为 Coze 插件优化）

    所有字段扁平返回，避免嵌套 Object/Array 导致 Coze 解析问题。
    Coze 插件只需配置 String 和 Number 类型的输出参数。

    - **symbol**: 股票代码（如 AAPL, 00700.HK）
    - **market**: 市场（us/hk/cn）
    """
    if symbol == "auto" or not symbol:
        symbol = "AAPL"
    if market == "auto" or not market:
        market = "us"

    symbol, market = normalize_stock_symbol(symbol, market)

    try:
        info, data = fetch_yf_data(symbol)

        if data.empty:
            raise HTTPException(status_code=404, detail="未找到股票数据")

        signal_data = get_trading_signal(data, symbol)
        trade_points = detect_trade_points(data, symbol)
        indicators = signal_data["indicators"]

        current_price = round(data['Close'].iloc[-1], 2)
        prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else current_price
        change_percent = round((current_price - prev_close) / prev_close * 100, 2) if prev_close != 0 else 0

        # 最近5个交易日K线，拼成一个字符串
        kline_text_lines = []
        for index, row in data.tail(5).iterrows():
            kline_text_lines.append(
                f"{index.strftime('%Y-%m-%d')} 开{round(row['Open'],2)} "
                f"高{round(row['High'],2)} 低{round(row['Low'],2)} "
                f"收{round(row['Close'],2)} 量{int(row['Volume'])}"
            )

        # 信号列表拼成一个字符串
        signals_text = "；".join(signal_data["signals"]) if signal_data["signals"] else "无明显信号"

        # 买卖点文字
        trade_point_cn = {
            "strong_buy": "强烈买入",
            "buy": "建议买入",
            "sell": "建议卖出",
            "strong_sell": "强烈卖出",
            "hold": "观望等待"
        }
        buy_reasons_text = "；".join(trade_points["buy_reasons"]) if trade_points["buy_reasons"] else ""
        sell_reasons_text = "；".join(trade_points["sell_reasons"]) if trade_points["sell_reasons"] else ""

        return {
            # 基础信息
            "symbol": str(symbol),
            "name": str(info.get("longName", "N/A")),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": str(info.get("currency", "USD")),
            "market": str(market),
            "analysis_time": datetime.now().isoformat(),
            # 买卖信号
            "signal": str(signal_data["signal"]),
            "confidence": str(signal_data["confidence"]),
            "key_signals_text": signals_text,
            # 买卖点（V5新增）
            "trade_point": str(trade_points["trade_point"]),
            "trade_point_cn": trade_point_cn.get(trade_points["trade_point"], "观望"),
            "trade_score": trade_points["score"],
            "buy_reasons_text": buy_reasons_text,
            "sell_reasons_text": sell_reasons_text,
            "entry_price": trade_points["entry_price"],
            "stop_loss": trade_points["stop_loss"],
            "take_profit": trade_points["take_profit"],
            # 技术指标（全部扁平化）
            "rsi": round(indicators["rsi"], 2),
            "rsi_prev": round(indicators["rsi_prev"], 2),
            "rsi_delta": indicators["rsi_delta"],
            "macd_value": round(indicators["macd"]["macd"], 4),
            "macd_signal": round(indicators["macd"]["signal"], 4),
            "macd_histogram": round(indicators["macd"]["histogram"], 4),
            "macd_cross": str("golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none")),
            "kdj_k": round(indicators["kdj"]["k"], 2),
            "kdj_d": round(indicators["kdj"]["d"], 2),
            "kdj_j": round(indicators["kdj"]["j"], 2),
            "boll_upper": round(indicators["bollinger_bands"]["upper"], 2),
            "boll_middle": round(indicators["bollinger_bands"]["middle"], 2),
            "boll_lower": round(indicators["bollinger_bands"]["lower"], 2),
            "ma5": round(indicators["ma5"], 2),
            "ma10": round(indicators["ma10"], 2),
            "ma20": round(indicators["ma20"], 2),
            "ma50": round(indicators["ma50"], 2) if indicators["ma50"] else 0,
            # 股票信息（扁平化）
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": round(info.get("trailingPE", 0), 2),
            "week52_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "week52_low": round(info.get("fiftyTwoWeekLow", 0), 2),
            "volume": info.get("volume", 0),
            # 新增：成交量、趋势、支撑阻力
            "volume_signal": str(signal_data["volume_signal"]),
            "volume_ratio": signal_data["volume_ratio"],
            "trend_direction": str(signal_data["trend_direction"]),
            "support_level": signal_data["support_level"],
            "resistance_level": signal_data["resistance_level"],
            # K线（文本格式）
            "kline_text": "\n".join(kline_text_lines),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析股票失败: {str(e)}")

# ==================== V3: 多股对比 & 加密货币 ====================

@app.get("/stock/compare")
def compare_stocks(symbols: str = "AAPL,MSFT,GOOG", market: str = "us"):
    """
    多股对比分析接口（专为 Coze 插件优化，扁平化返回）

    传入多个股票代码（逗号分隔），返回每只股票的核心指标对比。
    最多支持5只股票同时对比。

    - **symbols**: 股票代码，逗号分隔（如 "AAPL,MSFT,GOOG"）
    - **market**: 市场（us/hk）
    """
    if market == "auto" or not market:
        market = "us"

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(symbol_list) > 5:
            symbol_list = symbol_list[:5]
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="至少需要2只股票进行对比")

        # 标准化股票代码
        symbol_list = [normalize_stock_symbol(s, market)[0] for s in symbol_list]

        results = []
        for sym in symbol_list:
            try:
                info, data = fetch_yf_data(sym, period="6mo")

                if data.empty:
                    results.append({
                        "symbol": sym,
                        "name": sym,
                        "status": "not_found"
                    })
                    continue

                signal_data = get_trading_signal(data, sym)
                indicators = signal_data["indicators"]

                current_price = round(data['Close'].iloc[-1], 2)
                prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else current_price
                change_percent = round((current_price - prev_close) / prev_close * 100, 2) if prev_close != 0 else 0

                results.append({
                    "symbol": sym,
                    "name": str(info.get("longName", sym)),
                    "current_price": current_price,
                    "change_percent": change_percent,
                    "signal": str(signal_data["signal"]),
                    "confidence": str(signal_data["confidence"]),
                    "rsi": round(indicators["rsi"], 2),
                    "macd_cross": str("golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none")),
                    "macd_histogram": round(indicators["macd"]["histogram"], 4),
                    "kdj_k": round(indicators["kdj"]["k"], 2),
                    "kdj_d": round(indicators["kdj"]["d"], 2),
                    "volume_ratio": signal_data["volume_ratio"],
                    "trend_direction": str(signal_data["trend_direction"]),
                    "support_level": signal_data["support_level"],
                    "resistance_level": signal_data["resistance_level"],
                    "status": "ok"
                })
            except Exception as e:
                results.append({
                    "symbol": sym,
                    "name": sym,
                    "status": "error",
                    "error": str(e)
                })

        # 计算对比维度：谁最强/最弱
        valid_results = [r for r in results if r["status"] == "ok"]
        summary = {}
        if valid_results:
            # RSI最低的（最接近超卖，可能反弹机会）
            rsi_sorted = sorted(valid_results, key=lambda x: x["rsi"])
            summary["rsi_lowest"] = {"symbol": rsi_sorted[0]["symbol"], "rsi": rsi_sorted[0]["rsi"]}
            # RSI最高的（最接近超买，回调风险最大）
            summary["rsi_highest"] = {"symbol": rsi_sorted[-1]["symbol"], "rsi": rsi_sorted[-1]["rsi"]}
            # 涨幅最大
            change_sorted = sorted(valid_results, key=lambda x: x["change_percent"], reverse=True)
            summary["best_performer"] = {"symbol": change_sorted[0]["symbol"], "change": change_sorted[0]["change_percent"]}
            # 跌幅最大
            summary["worst_performer"] = {"symbol": change_sorted[-1]["symbol"], "change": change_sorted[-1]["change_percent"]}

        return {
            "market": market,
            "total": len(results),
            "success": len(valid_results),
            "stocks_text": json.dumps(results, ensure_ascii=False),
            "summary_text": json.dumps(summary, ensure_ascii=False),
            "analysis_time": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"多股对比失败: {str(e)}")


@app.get("/crypto/analyze")
def analyze_crypto(symbol: str = "BTC-USD"):
    """
    加密货币分析接口（扁平化，适配 Coze 插件）

    复用现有技术指标计算逻辑，支持比特币、以太坊等主流加密货币。
    yfinance 格式：BTC-USD, ETH-USD, BNB-USD 等。

    - **symbol**: 加密货币代码（如 BTC-USD, ETH-USD）
    """
    if not symbol:
        symbol = "BTC-USD"
    # 自动补全 -USD 后缀
    symbol_upper = symbol.upper()
    if not symbol_upper.endswith("-USD"):
        symbol = symbol_upper + "-USD"

    try:
        info, data = fetch_yf_data(symbol, period="6mo")

        if data.empty:
            raise HTTPException(status_code=404, detail=f"未找到加密货币数据: {symbol}")

        signal_data = get_trading_signal(data, symbol)
        indicators = signal_data["indicators"]

        current_price = round(data['Close'].iloc[-1], 2)
        prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else current_price
        change_percent = round((current_price - prev_close) / prev_close * 100, 2) if prev_close != 0 else 0

        # 最近5个交易日K线
        kline_text_lines = []
        for index, row in data.tail(5).iterrows():
            kline_text_lines.append(
                f"{index.strftime('%Y-%m-%d')} 开{round(row['Open'],2)} "
                f"高{round(row['High'],2)} 低{round(row['Low'],2)} "
                f"收{round(row['Close'],2)} 量{int(row['Volume'])}"
            )

        signals_text = "；".join(signal_data["signals"]) if signal_data["signals"] else "无明显信号"

        # 提取币种简称
        coin_name = symbol.replace("-USD", "")

        return {
            # 基础信息
            "symbol": str(coin_name),
            "name": str(info.get("shortName", coin_name)),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": "USD",
            "asset_type": "crypto",
            "analysis_time": datetime.now().isoformat(),
            # 买卖信号
            "signal": str(signal_data["signal"]),
            "confidence": str(signal_data["confidence"]),
            "key_signals_text": signals_text,
            # 技术指标（扁平化）
            "rsi": round(indicators["rsi"], 2),
            "rsi_prev": round(indicators["rsi_prev"], 2),
            "rsi_delta": indicators["rsi_delta"],
            "macd_value": round(indicators["macd"]["macd"], 4),
            "macd_signal": round(indicators["macd"]["signal"], 4),
            "macd_histogram": round(indicators["macd"]["histogram"], 4),
            "macd_cross": str("golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none")),
            "kdj_k": round(indicators["kdj"]["k"], 2),
            "kdj_d": round(indicators["kdj"]["d"], 2),
            "kdj_j": round(indicators["kdj"]["j"], 2),
            "boll_upper": round(indicators["bollinger_bands"]["upper"], 2),
            "boll_middle": round(indicators["bollinger_bands"]["middle"], 2),
            "boll_lower": round(indicators["bollinger_bands"]["lower"], 2),
            "ma5": round(indicators["ma5"], 2),
            "ma10": round(indicators["ma10"], 2),
            "ma20": round(indicators["ma20"], 2),
            "ma50": round(indicators["ma50"], 2) if indicators["ma50"] else 0,
            # 市场信息
            "market_cap": info.get("marketCap", 0),
            "volume_24h": info.get("volume", 0),
            # 成交量、趋势、支撑阻力
            "volume_signal": str(signal_data["volume_signal"]),
            "volume_ratio": signal_data["volume_ratio"],
            "trend_direction": str(signal_data["trend_direction"]),
            "support_level": signal_data["support_level"],
            "resistance_level": signal_data["resistance_level"],
            # K线（文本格式）
            "kline_text": "\n".join(kline_text_lines),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加密货币分析失败: {str(e)}")


# ==================== V4: 汇率分析 ====================

# 支持的汇率对映射（用户友好名称 → yfinance 代码）
FOREX_PAIRS = {
    "USDCNY": "CNY=X",       # 美元/人民币
    "USDJPY": "JPY=X",       # 美元/日元
    "USDEUR": "EURUSD=X",    # 美元/欧元（反向报价）
    "USDGBP": "GBPUSD=X",    # 美元/英镑（反向报价）
    "USDKRW": "KRW=X",       # 美元/韩元
    "USDHKD": "HKD=X",       # 美元/港币
    "USDSGD": "SGD=X",       # 美元/新加坡元
    "USDTWD": "TWD=X",       # 美元/新台币
    "USDINR": "INR=X",       # 美元/印度卢比
}

FOREX_NAMES = {
    "CNY=X": "美元/人民币",
    "JPY=X": "美元/日元",
    "EURUSD=X": "欧元/美元",
    "GBPUSD=X": "英镑/美元",
    "KRW=X": "美元/韩元",
    "HKD=X": "美元/港币",
    "SGD=X": "美元/新加坡元",
    "TWD=X": "美元/新台币",
    "INR=X": "美元/印度卢比",
}

# 反向报价的货币对（1欧元=?美元，而不是1美元=?欧元）
REVERSED_PAIRS = {"EURUSD=X", "GBPUSD=X"}


@app.get("/forex/analyze")
def analyze_forex(pair: str = "USDCNY"):
    """
    汇率技术分析接口（扁平化，适配 Coze 插件）

    支持主流汇率对的技术分析，复用现有技术指标计算逻辑。
    yfinance 数据源，每日更新。

    - **pair**: 汇率对代码，如 USDCNY、USDJPY、USDEUR、USDGBP、USDKRW、USDHKD
    """
    pair_upper = pair.upper().strip()

    # 查找 yfinance 代码
    yf_symbol = FOREX_PAIRS.get(pair_upper)
    if not yf_symbol:
        # 尝试直接作为 yfinance 代码使用
        yf_symbol = pair_upper

    try:
        _, data = fetch_yf_data(yf_symbol, period="6mo")

        if data.empty:
            raise HTTPException(status_code=404, detail=f"未找到汇率数据: {pair}（yfinance代码: {yf_symbol}）")

        # 复用技术指标计算
        signal_data = get_trading_signal(data, pair)
        indicators = signal_data["indicators"]

        current_price = round(data['Close'].iloc[-1], 4)
        prev_close = round(data['Close'].iloc[-2], 4) if len(data) > 1 else current_price
        change_percent = round((current_price - prev_close) / prev_close * 100, 4) if prev_close != 0 else 0

        # 最近5个交易日K线
        kline_text_lines = []
        for index, row in data.tail(5).iterrows():
            kline_text_lines.append(
                f"{index.strftime('%Y-%m-%d')} 开{round(row['Open'],4)} "
                f"高{round(row['High'],4)} 低{round(row['Low'],4)} "
                f"收{round(row['Close'],4)} 量{int(row['Volume'])}"
            )

        signals_text = "；".join(signal_data["signals"]) if signal_data["signals"] else "无明显信号"

        # 获取汇率对中文名称
        pair_name = FOREX_NAMES.get(yf_symbol, pair)

        # 计算近期波动率（20日标准差/均值）
        returns = data['Close'].pct_change().dropna()
        volatility_20d = round(returns.tail(20).std() * 100, 2) if len(returns) >= 20 else 0

        # 计算N日最高最低（支撑阻力参考）
        recent_high = round(data['High'].tail(20).max(), 4)
        recent_low = round(data['Low'].tail(20).min(), 4)

        return {
            # 基础信息
            "pair": str(pair_upper),
            "name": str(pair_name),
            "current_price": current_price,
            "change_percent": change_percent,
            "volatility_20d": volatility_20d,
            "is_reversed": yf_symbol in REVERSED_PAIRS,
            "asset_type": "forex",
            "analysis_time": datetime.now().isoformat(),
            # 买卖信号
            "signal": str(signal_data["signal"]),
            "confidence": str(signal_data["confidence"]),
            "key_signals_text": signals_text,
            # 技术指标（全部扁平化）
            "rsi": round(indicators["rsi"], 2),
            "rsi_prev": round(indicators["rsi_prev"], 2),
            "rsi_delta": indicators["rsi_delta"],
            "macd_value": round(indicators["macd"]["macd"], 4),
            "macd_signal": round(indicators["macd"]["signal"], 4),
            "macd_histogram": round(indicators["macd"]["histogram"], 4),
            "macd_cross": str("golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none")),
            "kdj_k": round(indicators["kdj"]["k"], 2),
            "kdj_d": round(indicators["kdj"]["d"], 2),
            "kdj_j": round(indicators["kdj"]["j"], 2),
            "boll_upper": round(indicators["bollinger_bands"]["upper"], 4),
            "boll_middle": round(indicators["bollinger_bands"]["middle"], 4),
            "boll_lower": round(indicators["bollinger_bands"]["lower"], 4),
            "ma5": round(indicators["ma5"], 4),
            "ma10": round(indicators["ma10"], 4),
            "ma20": round(indicators["ma20"], 4),
            "ma50": round(indicators["ma50"], 4) if indicators["ma50"] else 0,
            # 成交量、趋势、支撑阻力
            "volume_signal": str(signal_data["volume_signal"]),
            "volume_ratio": signal_data["volume_ratio"],
            "trend_direction": str(signal_data["trend_direction"]),
            "support_level": recent_low,
            "resistance_level": recent_high,
            # K线（文本格式）
            "kline_text": "\n".join(kline_text_lines),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"汇率分析失败: {str(e)}")


def detect_trade_points(data, symbol):
    """
    精准买卖点检测（V5新增）

    在 get_trading_signal 基础上，组合多个指标识别高胜率买卖时机。

    买入点识别规则（需满足2个及以上条件才算 strong_buy）：
    1. MACD 金叉（DIF上穿DEA）
    2. RSI 从超卖区回升（前日<30，今日≥30）
    3. KDJ 超卖金叉（J<0 且 K上穿D）
    4. 放量上涨（量比>1.5 + 阳线）
    5. 价格触及布林带下轨后反弹（今日收>开）
    6. 均线多头排列 + 价格回踩MA20支撑

    卖出点识别规则（需满足2个及以上条件才算 strong_sell）：
    1. MACD 死叉（DIF下穿DEA）
    2. RSI 从超买区回落（前日>70，今日≤70）
    3. KDJ 超买卖叉（J>100 且 K下穿D）
    4. 放量下跌（量比>1.5 + 阴线）
    5. 价格触及布林带上轨后回落（今日收<开）
    6. 均线空头排列 + 价格反弹至MA20阻力

    返回：
    - trade_point: strong_buy / buy / sell / strong_sell / hold
    - buy_reasons: 买入理由列表
    - sell_reasons: 卖出理由列表
    - entry_price: 建议入场价（买入点）
    - stop_loss: 建议止损价
    - take_profit: 建议止盈价
    - score: 综合评分 (-10 ~ +10)
    """
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    is_green = current_price > data['Open'].iloc[-1] if len(data) > 1 else True

    rsi, rsi_prev, rsi_delta = calculate_rsi(data)
    macd_data = calculate_macd(data)
    kdj_data = calculate_kdj(data)
    boll = calculate_bollinger_bands(data)
    vol_status, vol_ratio = calculate_volume_signal(data)

    ma5 = data['Close'].rolling(window=5).mean().iloc[-1]
    ma10 = data['Close'].rolling(window=10).mean().iloc[-1]
    ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
    ma50 = data['Close'].rolling(window=50).mean().iloc[-1] if len(data) >= 50 else None

    buy_reasons = []
    sell_reasons = []
    buy_count = 0
    sell_count = 0

    # ========== 买入点检测 ==========

    # 1. MACD 金叉
    if macd_data['golden_cross']:
        buy_reasons.append(f"MACD金叉确认（DIF={round(macd_data['macd'],4)}，DEA={round(macd_data['signal'],4)}）")
        buy_count += 1

    # 2. RSI 从超卖区回升
    if rsi_prev < 30 and rsi >= 30:
        buy_reasons.append(f"RSI脱离超卖区（{round(rsi_prev,1)}→{round(rsi,1)}），反转信号")
        buy_count += 1
    elif rsi < 20:
        buy_reasons.append(f"RSI极度超卖（{round(rsi,1)}），超跌反弹概率大")
        buy_count += 1

    # 3. KDJ 超卖金叉（J<0 区间 K上穿D）
    k, d, j = kdj_data['k'], kdj_data['d'], kdj_data['j']
    if len(data) >= 3:
        # 计算前一日 KDJ
        prev_rsv = (data['Close'].iloc[-3] - data['Low'].iloc[-12:-3].min()) / \
                   (data['High'].iloc[-12:-3].max() - data['Low'].iloc[-12:-3].min()) * 100 \
                   if len(data) >= 12 else 50
        prev_k_val = (prev_rsv + 2 * k) / 3  # 近似
        kdj_golden = (prev_k_val < d and k >= d)  # K 从下穿越 D
    else:
        kdj_golden = False

    if j < 0 and kdj_golden:
        buy_reasons.append(f"KDJ超卖区金叉（K={k}，D={d}，J={round(j,1)}）")
        buy_count += 1
    elif j < 0:
        buy_reasons.append(f"KDJ的J值深度超卖（{round(j,1)}），反弹在即")
        buy_count += 0.5

    # 4. 放量上涨
    if vol_status in ("high_volume", "above_avg") and is_green:
        buy_reasons.append(f"放量上涨（量比{vol_ratio}倍），资金进场确认")
        buy_count += 1

    # 5. 触及布林带下轨后反弹
    prev_low = data['Low'].iloc[-2] if len(data) > 1 else current_price
    if prev_low <= boll['lower'] and is_green:
        buy_reasons.append(f"触及布林带下轨（{round(boll['lower'],2)}）后反弹，支撑有效")
        buy_count += 1
    elif current_price <= boll['lower'] * 1.01 and is_green:
        buy_reasons.append(f"接近布林带下轨后反弹，支撑区企稳")
        buy_count += 0.5

    # 6. 均线多头 + 回踩MA20
    ma_bullish = current_price > ma5 and ma5 > ma10 and ma10 > ma20
    if ma_bullish and prev_close <= ma20 and current_price > ma20:
        buy_reasons.append(f"多头趋势回踩MA20（{round(ma20,2)}）支撑后企稳")
        buy_count += 1

    # ========== 卖出点检测 ==========

    # 1. MACD 死叉
    if macd_data['death_cross']:
        sell_reasons.append(f"MACD死叉确认（DIF={round(macd_data['macd'],4)}，DEA={round(macd_data['signal'],4)}）")
        sell_count += 1

    # 2. RSI 从超买区回落
    if rsi_prev > 70 and rsi <= 70:
        sell_reasons.append(f"RSI脱离超买区（{round(rsi_prev,1)}→{round(rsi,1)}），见顶信号")
        sell_count += 1
    elif rsi > 85:
        sell_reasons.append(f"RSI极度超买（{round(rsi,1)}），随时可能回调")
        sell_count += 1

    # 3. KDJ 超买卖叉（J>100 区间 K下穿D）
    if j > 100 and not kdj_golden and k < d:
        sell_reasons.append(f"KDJ超买区死叉（K={k}，D={d}，J={round(j,1)}）")
        sell_count += 1
    elif j > 100:
        sell_reasons.append(f"KDJ的J值深度超买（{round(j,1)}），短期风险极大")
        sell_count += 0.5

    # 4. 放量下跌
    if vol_status in ("high_volume", "above_avg") and not is_green:
        sell_reasons.append(f"放量下跌（量比{vol_ratio}倍），资金出逃确认")
        sell_count += 1

    # 5. 触及布林带上轨后回落
    prev_high = data['High'].iloc[-2] if len(data) > 1 else current_price
    if prev_high >= boll['upper'] and not is_green:
        sell_reasons.append(f"触及布林带上轨（{round(boll['upper'],2)}）后回落，压力有效")
        sell_count += 1
    elif current_price >= boll['upper'] * 0.99 and not is_green:
        sell_reasons.append(f"接近布林带上轨后回落，压力区受阻")
        sell_count += 0.5

    # 6. 均线空头 + 反弹至MA20
    ma_bearish = current_price < ma5 and ma5 < ma10 and ma10 < ma20
    if ma_bearish and prev_close >= ma20 and current_price < ma20:
        sell_reasons.append(f"空头趋势反弹至MA20（{round(ma20,2)}）后继续下跌")
        sell_count += 1

    # ========== 综合判断 ==========

    # 评分：买入+卖出互相抵消
    score = round((buy_count - sell_count) * 2, 1)

    if buy_count >= 2 and sell_count == 0:
        trade_point = "strong_buy"
    elif buy_count >= 1.5 and buy_count > sell_count:
        trade_point = "buy"
    elif sell_count >= 2 and buy_count == 0:
        trade_point = "strong_sell"
    elif sell_count >= 1.5 and sell_count > buy_count:
        trade_point = "sell"
    else:
        trade_point = "hold"

    # 建议价格（基于近期高低点）
    recent_low = data['Low'].tail(20).min()
    recent_high = data['High'].tail(20).max()
    atr = (data['High'].tail(14).max() - data['Low'].tail(14).min()) / 14  # 简化ATR

    if trade_point in ("strong_buy", "buy"):
        entry_price = round(current_price * 0.995, 2)   # 稍低于当前价
        stop_loss = round(recent_low * 0.98, 2)          # 近期低点下方2%
        take_profit = round(current_price + atr * 3, 2)  # 3倍ATR
    elif trade_point in ("strong_sell", "sell"):
        entry_price = 0  # 卖出不需要入场价
        stop_loss = 0
        take_profit = round(recent_low * 1.02, 2)       # 回落到近期低点附近
    else:
        entry_price = 0
        stop_loss = round(recent_low * 0.97, 2)
        take_profit = round(recent_high * 1.03, 2)

    return {
        "trade_point": trade_point,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "score": score,
    }


def normalize_stock_symbol(symbol: str, market: str = "us") -> tuple:
    """
    标准化股票代码，自动补全市场后缀

    - 港股(hk)：纯数字代码自动补 .HK（如 0700 → 0700.HK, 00700 → 00700.HK）
    - 美股(us)：不做处理（yfinance 直接支持）
    - A股(cn)：补 .SS（上交所）或 .SZ（深交所），暂不自动区分
    - 已有后缀的代码直接返回

    returns: (normalized_symbol, detected_market)
    """
    sym = symbol.strip().upper()

    # 已有后缀，直接返回
    if sym.endswith(".HK") or sym.endswith(".SS") or sym.endswith(".SZ"):
        detected = "hk" if sym.endswith(".HK") else "cn"
        return sym, detected

    # 港股：纯数字（3-5位）→ 补 .HK
    if market.lower() == "hk" or (sym.isdigit() and 3 <= len(sym) <= 5):
        return f"{sym}.HK", "hk"

    # A股：6开头上交所(.SS)，0/3开头深交所(.SZ)
    if market.lower() == "cn" and len(sym) == 6 and sym.isdigit():
        if sym.startswith("6"):
            return f"{sym}.SS", "cn"
        elif sym.startswith("0") or sym.startswith("3"):
            return f"{sym}.SZ", "cn"

    return sym, market.lower()


# ==================== V5: 买卖点检测 & 批量扫描 ====================

# 默认美股扫描列表（科技蓝筹）
DEFAULT_US_SCAN = "AAPL,MSFT,GOOG,AMZN,NVDA,TSLA,META,NFLX,AMD,INTC"
# 默认港股扫描列表
DEFAULT_HK_SCAN = "0700,9988,1810,3690,9999,2318,1299,0388,0981,1211"
# 默认A股扫描列表
DEFAULT_CN_SCAN = "600519,000858,300750,601318,000001,600036,002410,601899,600900,300059"


@app.get("/stock/scan")
def scan_stocks(
    symbols: str = "",
    market: str = "us",
    min_score: float = 3.0
):
    """
    批量扫描买卖点接口（V5新增，适配 Coze 插件，扁平化返回）

    扫描一篮子股票，识别当前有明确买卖点的标的，按信号强度排序。
    支持自定义股票列表或使用默认热门列表。

    - **symbols**: 股票代码，逗号分隔（留空使用默认列表）
    - **market**: 市场（us/hk/cn）
    - **min_score**: 最低信号分数（默认3.0，只返回评分≥此值的标的）
    """
    if market == "auto" or not market:
        market = "us"

    # 如果没传 symbols，使用默认列表
    if not symbols or symbols.strip() == "":
        if market.lower() == "hk":
            symbols = DEFAULT_HK_SCAN
        elif market.lower() == "cn":
            symbols = DEFAULT_CN_SCAN
        else:
            symbols = DEFAULT_US_SCAN

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(symbol_list) > 20:
            symbol_list = symbol_list[:20]

        # 标准化股票代码
        normalized = [normalize_stock_symbol(s, market) for s in symbol_list]

        results = []
        failed_count = 0
        last_error = ""
        for sym, detected_market in normalized:
            try:
                info, data = fetch_yf_data(sym)

                if data.empty:
                    continue

                signal_data = get_trading_signal(data, sym)
                trade_points = detect_trade_points(data, sym)

                # 过滤掉 HOLD 信号且分数不够的
                if trade_points["trade_point"] == "hold" and abs(trade_points["score"]) < min_score:
                    continue

                current_price = round(data['Close'].iloc[-1], 2)
                prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else current_price
                change_percent = round((current_price - prev_close) / prev_close * 100, 2) if prev_close != 0 else 0

                name = str(info.get("longName", info.get("shortName", sym)))

                results.append({
                    "symbol": str(sym),
                    "name": name,
                    "current_price": current_price,
                    "change_percent": change_percent,
                    "trade_point": str(trade_points["trade_point"]),
                    "score": trade_points["score"],
                    "entry_price": trade_points["entry_price"],
                    "stop_loss": trade_points["stop_loss"],
                    "take_profit": trade_points["take_profit"],
                    "buy_reasons_text": "；".join(trade_points["buy_reasons"]) if trade_points["buy_reasons"] else "",
                    "sell_reasons_text": "；".join(trade_points["sell_reasons"]) if trade_points["sell_reasons"] else "",
                    "rsi": round(signal_data["indicators"]["rsi"], 2),
                    "macd_cross": str("golden" if signal_data["indicators"]["macd"]["golden_cross"] else ("death" if signal_data["indicators"]["macd"]["death_cross"] else "none")),
                    "volume_ratio": signal_data["volume_ratio"],
                    "trend_direction": str(signal_data["trend_direction"]),
                    "status": "ok"
                })
            except Exception as e:
                failed_count += 1
                last_error = str(e)
                continue

        # 按评分绝对值排序（最强的信号排前面）
        results.sort(key=lambda x: abs(x["score"]), reverse=True)

        # 分类统计
        buy_stocks = [r for r in results if r["trade_point"] in ("strong_buy", "buy")]
        sell_stocks = [r for r in results if r["trade_point"] in ("strong_sell", "sell")]

        return {
            "market": market,
            "total_scanned": len(symbol_list),
            "total_signals": len(results),
            "failed_count": failed_count,
            "last_error": last_error if failed_count > 0 else "",
            "buy_count": len(buy_stocks),
            "sell_count": len(sell_stocks),
            "top_buy": buy_stocks[:3] if buy_stocks else [],
            "top_sell": sell_stocks[:3] if sell_stocks else [],
            "all_signals": results,
            "scan_time": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量扫描失败: {str(e)}")


@app.get("/stock/tradepoint")
def get_trade_point_flat(symbol: str = "AAPL", market: str = "us"):
    """
    单股精准买卖点接口（V5新增，扁平化，适配 Coze 插件）

    专精识别当前是否处于最佳买入/卖出时点。
    组合MACD、RSI、KDJ、布林带、均线、成交量六大维度，
    输出明确的买卖点类型、触发原因和价格建议。

    - **symbol**: 股票代码（如 AAPL, 0700, 002410）
    - **market**: 市场（us/hk/cn）
    """
    if symbol == "auto" or not symbol:
        symbol = "AAPL"
    if market == "auto" or not market:
        market = "us"

    symbol, market = normalize_stock_symbol(symbol, market)

    try:
        info, data = fetch_yf_data(symbol)

        if data.empty:
            raise HTTPException(status_code=404, detail="未找到股票数据")

        signal_data = get_trading_signal(data, symbol)
        trade_points = detect_trade_points(data, symbol)
        indicators = signal_data["indicators"]

        current_price = round(data['Close'].iloc[-1], 2)
        prev_close = round(data['Close'].iloc[-2], 2) if len(data) > 1 else current_price
        change_percent = round((current_price - prev_close) / prev_close * 100, 2) if prev_close != 0 else 0

        # 买卖点类型转中文
        trade_point_cn = {
            "strong_buy": "强烈买入",
            "buy": "建议买入",
            "sell": "建议卖出",
            "strong_sell": "强烈卖出",
            "hold": "观望等待"
        }

        # 最近5个交易日K线
        kline_text_lines = []
        for index, row in data.tail(5).iterrows():
            kline_text_lines.append(
                f"{index.strftime('%Y-%m-%d')} 开{round(row['Open'],2)} "
                f"高{round(row['High'],2)} 低{round(row['Low'],2)} "
                f"收{round(row['Close'],2)} 量{int(row['Volume'])}"
            )

        return {
            # 基础信息
            "symbol": str(symbol),
            "name": str(info.get("longName", "N/A")),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": str(info.get("currency", "USD")),
            "market": str(market),
            "analysis_time": datetime.now().isoformat(),
            # 买卖点核心
            "trade_point": str(trade_points["trade_point"]),
            "trade_point_cn": trade_point_cn.get(trade_points["trade_point"], "观望"),
            "score": trade_points["score"],
            "buy_reasons_text": "；".join(trade_points["buy_reasons"]) if trade_points["buy_reasons"] else "暂无买入理由",
            "sell_reasons_text": "；".join(trade_points["sell_reasons"]) if trade_points["sell_reasons"] else "暂无卖出理由",
            # 价格建议
            "entry_price": trade_points["entry_price"],
            "stop_loss": trade_points["stop_loss"],
            "take_profit": trade_points["take_profit"],
            # 核心指标（扁平化）
            "rsi": round(indicators["rsi"], 2),
            "rsi_prev": round(indicators["rsi_prev"], 2),
            "macd_cross": str("golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none")),
            "macd_histogram": round(indicators["macd"]["histogram"], 4),
            "kdj_k": round(indicators["kdj"]["k"], 2),
            "kdj_d": round(indicators["kdj"]["d"], 2),
            "kdj_j": round(indicators["kdj"]["j"], 2),
            # 辅助指标
            "volume_ratio": signal_data["volume_ratio"],
            "trend_direction": str(signal_data["trend_direction"]),
            "support_level": signal_data["support_level"],
            "resistance_level": signal_data["resistance_level"],
            # K线
            "kline_text": "\n".join(kline_text_lines),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"买卖点分析失败: {str(e)}")


# ===== 双色球玄学号码映射 API =====

# 天干号码映射（红球1-33）
_TIANGAN_MAP = {
    "甲": {"wuxing": "木", "red_balls": [1, 11, 21, 31]},
    "乙": {"wuxing": "木", "red_balls": [2, 12, 22, 32]},
    "丙": {"wuxing": "火", "red_balls": [3, 13, 23, 33]},
    "丁": {"wuxing": "火", "red_balls": [4, 14, 24]},
    "戊": {"wuxing": "土", "red_balls": [5, 15, 25]},
    "己": {"wuxing": "土", "red_balls": [6, 16, 26]},
    "庚": {"wuxing": "金", "red_balls": [7, 17, 27]},
    "辛": {"wuxing": "金", "red_balls": [8, 18, 28]},
    "壬": {"wuxing": "水", "red_balls": [9, 19, 29]},
    "癸": {"wuxing": "水", "red_balls": [10, 20, 30]},
}

# 地支号码映射（红球1-33）
_DIZHI_RED_MAP = {
    "子": {"wuxing": "水", "red_balls": [1, 13, 25]},
    "丑": {"wuxing": "土", "red_balls": [2, 14, 26]},
    "寅": {"wuxing": "木", "red_balls": [3, 15, 27]},
    "卯": {"wuxing": "木", "red_balls": [4, 16, 28]},
    "辰": {"wuxing": "土", "red_balls": [5, 17, 29]},
    "巳": {"wuxing": "火", "red_balls": [6, 18, 30]},
    "午": {"wuxing": "火", "red_balls": [7, 19, 31]},
    "未": {"wuxing": "土", "red_balls": [8, 20, 32]},
    "申": {"wuxing": "金", "red_balls": [9, 21, 33]},
    "酉": {"wuxing": "金", "red_balls": [10, 22]},
    "戌": {"wuxing": "土", "red_balls": [11, 23]},
    "亥": {"wuxing": "水", "red_balls": [12, 24]},
}

# 地支号码映射（蓝球1-16）
_DIZHI_BLUE_MAP = {
    "子": 1, "丑": 2, "寅": 3, "卯": 4, "辰": 5, "巳": 6,
    "午": 7, "未": 8, "申": 9, "酉": 10, "戌": 11, "亥": 12,
}

# 五行号码总表（天干·地支·八卦综合映射）
_WUXING_MAP = {
    "金": {"red_balls": [7, 8, 9, 10, 17, 18, 25, 26], "blue_balls": [9, 10]},
    "木": {"red_balls": [1, 2, 3, 4, 15, 16, 23, 24, 31, 32], "blue_balls": [3, 4]},
    "水": {"red_balls": [7, 8, 19, 20, 27, 28, 33], "blue_balls": [1, 6, 12, 16]},
    "火": {"red_balls": [3, 4, 6, 11, 12, 18, 29, 30], "blue_balls": [7, 13]},
    "土": {"red_balls": [2, 5, 6, 13, 14, 21, 22, 27, 28], "blue_balls": [2, 5, 8, 11, 14, 15]},
}

# 日月水火映射表
_SUN_MOON_MAP = {
    "日": {"desc": "太阳·离卦·火·阳", "red_balls": [3, 4, 11, 29, 30], "blue_balls": [7]},
    "月": {"desc": "太阴·坎卦·水·阴", "red_balls": [7, 8, 19, 20, 27, 28], "blue_balls": [6]},
    "水": {"desc": "坎卦·壬癸·亥子", "red_balls": [7, 8, 19, 20, 27, 28, 33], "blue_balls": [1, 6]},
    "火": {"desc": "离卦·丙丁·巳午", "red_balls": [3, 4, 11, 12, 29, 30], "blue_balls": [7, 9]},
}

# 五行生克关系
_SHENGKE = {
    "相生": {"木生火": True, "火生土": True, "土生金": True, "金生水": True, "水生木": True},
    "相克": {"木克土": True, "土克水": True, "水克火": True, "火克金": True, "金克木": True},
}

# 月相判断
_MOON_PHASE = {
    1: "朔（新月）", 2: "朔后", 3: "朔后", 4: "朔后", 5: "上弦前",
    6: "上弦前", 7: "上弦前", 8: "上弦", 9: "上弦后", 10: "上弦后",
    11: "上弦后", 12: "望前", 13: "望前", 14: "望前", 15: "望（满月）",
    16: "望后", 17: "望后", 18: "望后", 19: "望后", 20: "下弦前",
    21: "下弦前", 22: "下弦前", 23: "下弦", 24: "下弦后", 25: "下弦后",
    26: "下弦后", 27: "下弦后", 28: "晦前", 29: "晦", 30: "晦",
}

# 月相红球号码映射（按月相阶段取象）
_MOON_PHASE_RED = {
    "朔（新月）": [1, 11, 21],
    "朔后": [2, 12, 22],
    "上弦前": [3, 13, 23],
    "上弦": [4, 14, 24],
    "上弦后": [5, 15, 25],
    "望前": [6, 16, 26],
    "望（满月）": [7, 17, 27],
    "望后": [8, 18, 28],
    "下弦前": [9, 19, 29],
    "下弦": [10, 20, 30],
    "下弦后": [11, 21, 31],
    "晦前": [12, 22, 32],
    "晦": [13, 23, 33],
}

# 月相蓝球号码映射（按月相阴阳消长）
_MOON_PHASE_BLUE = {
    "朔（新月）": [1],       # 极阴，坎水
    "朔后": [1, 2],          # 阴始消
    "上弦前": [2, 3],        # 阳渐长
    "上弦": [3, 4],          # 阴阳半
    "上弦后": [4, 5],        # 阳胜阴
    "望前": [5, 6],          # 阳将极
    "望（满月）": [6, 7],    # 极阳，离火
    "望后": [7, 8],          # 阳始消
    "下弦前": [8, 9],        # 阴渐长
    "下弦": [9, 10],         # 阴阳半
    "下弦后": [10, 11],      # 阴胜阳
    "晦前": [11, 12],        # 阴将极
    "晦": [12, 13],          # 极阴
}

# 月相吉凶倾向（纯娱乐）
_MOON_PHASE_LUCK = {
    "朔（新月）": "🌑 蛰伏期·宜守不宜攻·蓝球偏小号",
    "朔后": "🌱 萌动期·渐有转机·可小试",
    "上弦前": "🌿 生长中·阳气渐旺·偏红球中段",
    "上弦": "🌓 平衡期·阴阳各半·号码分散",
    "上弦后": "🌳 旺盛期·阳气充盈·红球偏大号",
    "望前": "🔥 将满期·能量蓄积·偏旺行号码",
    "望（满月）": "🌕 极盛期·阳气最旺·旺行+火行优先",
    "望后": "🌗 转衰期·盛极而衰·注意克我行号码",
    "下弦前": "🍂 收敛期·阳气渐退·偏生我行号码",
    "下弦": "🌓 平衡期·阴渐胜阳·注意泄行号码",
    "下弦后": "🌑 蛰伏前·阴气加重·蓝球偏小号",
    "晦前": "🕳️ 将晦期·能量最低·宜保守",
    "晦": "🌑 极暗期·最弱之时·蓝球取极小号",
}

# ===== v3.0 新增常量 =====

# 六十甲子纳音五行映射
_NAYIN_MAP = {
    "甲子": "海中金", "乙丑": "海中金", "丙寅": "炉中火", "丁卯": "炉中火",
    "戊辰": "大林木", "己巳": "大林木", "庚午": "路旁土", "辛未": "路旁土",
    "壬申": "剑锋金", "癸酉": "剑锋金", "甲戌": "山头火", "乙亥": "山头火",
    "丙子": "涧下水", "丁丑": "涧下水", "戊寅": "城头土", "己卯": "城头土",
    "庚辰": "白蜡金", "辛巳": "白蜡金", "壬午": "杨柳木", "癸未": "杨柳木",
    "甲申": "泉中水", "乙酉": "泉中水", "丙戌": "屋上土", "丁亥": "屋上土",
    "戊子": "霹雳火", "己丑": "霹雳火", "庚寅": "松柏木", "辛卯": "松柏木",
    "壬辰": "长流水", "癸巳": "长流水", "甲午": "沙中金", "乙未": "沙中金",
    "丙申": "山下火", "丁酉": "山下火", "戊戌": "平地木", "己亥": "平地木",
    "庚子": "壁上土", "辛丑": "壁上土", "壬寅": "金箔金", "癸卯": "金箔金",
    "甲辰": "覆灯火", "乙巳": "覆灯火", "丙午": "天河水", "丁未": "天河水",
    "戊申": "大驿土", "己酉": "大驿土", "庚戌": "钗钏金", "辛亥": "钗钏金",
    "壬子": "桑柘木", "癸丑": "桑柘木", "甲寅": "大溪水", "乙卯": "大溪水",
    "丙辰": "沙中土", "丁巳": "沙中土", "戊午": "天上火", "己未": "天上火",
    "庚申": "石榴木", "辛酉": "石榴木", "壬戌": "大海水", "癸亥": "大海水",
}

# 纳音五行提取（从纳音名称中提取五行属性）
_NAYIN_WUXING = {
    "海中金": "金", "炉中火": "火", "大林木": "木", "路旁土": "土", "剑锋金": "金",
    "山头火": "火", "涧下水": "水", "城头土": "土", "白蜡金": "金", "杨柳木": "木",
    "泉中水": "水", "屋上土": "土", "霹雳火": "火", "松柏木": "木", "长流水": "水",
    "沙中金": "金", "山下火": "火", "平地木": "木", "壁上土": "土", "金箔金": "金",
    "覆灯火": "火", "天河水": "水", "大驿土": "土", "钗钏金": "金", "桑柘木": "木",
    "大溪水": "水", "沙中土": "土", "天上火": "火", "石榴木": "木", "大海水": "水",
}

# 九宫飞星号码映射（每宫映射3个红球+1个蓝球）
# 按洛书九宫：1坎北、2坤西南、3震东、4巽东南、5中宫、6乾西北、7兑西、8艮东北、9离南
_JIUGONG_MAP = {
    1: {"name": "坎·北方", "red_balls": [1, 11, 21], "blue_balls": [1]},
    2: {"name": "坤·西南", "red_balls": [2, 12, 22], "blue_balls": [2]},
    3: {"name": "震·东方", "red_balls": [3, 13, 23], "blue_balls": [3]},
    4: {"name": "巽·东南", "red_balls": [4, 14, 24], "blue_balls": [4]},
    5: {"name": "中宫", "red_balls": [5, 15, 25], "blue_balls": [5]},
    6: {"name": "乾·西北", "red_balls": [6, 16, 26], "blue_balls": [6]},
    7: {"name": "兑·西方", "red_balls": [7, 17, 27], "blue_balls": [7]},
    8: {"name": "艮·东北", "red_balls": [8, 18, 28], "blue_balls": [8]},
    9: {"name": "离·南方", "red_balls": [9, 19, 29], "blue_balls": [9]},
}

# 飞星顺飞路径（中宫→乾→兑→艮→离→坎→坤→震→巽，即5→6→7→8→9→1→2→3→4）
_FEIXING_ORDER = [5, 6, 7, 8, 9, 1, 2, 3, 4]

# 飞星名称：1白、2黑、3碧、4绿、5黄、6白、7赤、8白、9紫
_FEIXING_NAMES = {
    1: "一白", 2: "二黑", 3: "三碧", 4: "四绿", 5: "五黄",
    6: "六白", 7: "七赤", 8: "八白", 9: "九紫",
}

# 年飞星入中宫计算：2024年=3碧入中，每减1年飞星+1（模9，0=9）
# 2024=3, 2025=2, 2026=1, 2027=9, 2028=8...
def _get_year_feixing(year: int) -> int:
    """计算年飞星入中宫的数字（1-9）"""
    # 公式：(11 - (year % 9)) % 9，0时取9
    star = (11 - (year % 9)) % 9
    return star if star != 0 else 9

# 地支→先天八卦方位映射
_DIZHI_BAGUA_MAP = {
    "子": {"gua": "坎", "fangwei": "北方", "red_balls": [1, 11, 21], "blue_balls": [1]},
    "丑": {"gua": "艮", "fangwei": "东北", "red_balls": [8, 18, 28], "blue_balls": [8]},
    "寅": {"gua": "艮", "fangwei": "东北", "red_balls": [8, 18, 28], "blue_balls": [8]},
    "卯": {"gua": "震", "fangwei": "东方", "red_balls": [3, 13, 23], "blue_balls": [3]},
    "辰": {"gua": "巽", "fangwei": "东南", "red_balls": [4, 14, 24], "blue_balls": [4]},
    "巳": {"gua": "巽", "fangwei": "东南", "red_balls": [4, 14, 24], "blue_balls": [4]},
    "午": {"gua": "离", "fangwei": "南方", "red_balls": [9, 19, 29], "blue_balls": [9]},
    "未": {"gua": "坤", "fangwei": "西南", "red_balls": [2, 12, 22], "blue_balls": [2]},
    "申": {"gua": "坤", "fangwei": "西南", "red_balls": [2, 12, 22], "blue_balls": [2]},
    "酉": {"gua": "兑", "fangwei": "西方", "red_balls": [7, 17, 27], "blue_balls": [7]},
    "戌": {"gua": "乾", "fangwei": "西北", "red_balls": [6, 16, 26], "blue_balls": [6]},
    "亥": {"gua": "乾", "fangwei": "西北", "red_balls": [6, 16, 26], "blue_balls": [6]},
}

# 时辰吉凶（纯娱乐）
_HOUR_LUCK = {
    "子": "🌃 夜半·阴极阳生·水行旺·蓝球偏小号",
    "丑": "🐄 鸡鸣·阴退阳进·土行暗旺·偏中号",
    "寅": "🐅 平旦·阳气初生·木行渐旺·偏大号",
    "卯": "🐇 日出·木行正旺·红球偏木行",
    "辰": "🐉 食时·土行旺·号码偏稳",
    "巳": "🐍 隅中·火行渐旺·红球偏火行",
    "午": "🐎 日中·火行极旺·旺行+火行优先",
    "未": "🐑 日昳·土行旺·偏生我行号码",
    "申": "🐒 晡时·金行渐旺·红球偏金行",
    "酉": "🐓 日入·金行正旺·蓝球偏金行",
    "戌": "🐕 黄昏·土行收·偏保守号码",
    "亥": "🐷 人定·水行旺·蓝球偏水行",
}


def _fmt(nums: list) -> str:
    """格式化号码列表为逗号分隔字符串，两位补零"""
    return ", ".join(f"{n:02d}" for n in nums)


def _get_shengke_info(day_wuxing: str) -> dict:
    """根据日柱五行获取生克关系"""
    sheng_order = ["木", "火", "土", "金", "水"]
    idx = sheng_order.index(day_wuxing)
    # 我生（泄）= 下一行，生我= 上一行，克我= 上一行的上一行，我克= 下一行的下一行
    sheng_wo = sheng_order[(idx - 1) % 5]   # 生我者
    wo_sheng = sheng_order[(idx + 1) % 5]   # 我生者（泄）
    ke_wo = sheng_order[(idx - 2) % 5]      # 克我者
    wo_ke = sheng_order[(idx + 2) % 5]      # 我克者
    return {
        "旺行": day_wuxing,
        "生我行": sheng_wo,
        "我生行(泄)": wo_sheng,
        "克我行": ke_wo,
        "我克行": wo_ke,
    }


@app.get("/ganzhi", tags=["双色球玄学映射"])
async def ganzhi_by_date(date: str = "2026-05-22", mode: str = "day_gan", hour_zhi: str = ""):
    """
    干支五行号码映射接口（按日期查询，专为Coze插件优化）
    �3.0：新增六柱干支号码、九宫飞星、纳音五行、八卦方位、时辰分析、热度升级。

    - **date**: 公历日期（格式：YYYY-MM-DD）
    - **mode**: 旺行判定逻辑，可选：
      - day_gan（默认）：日柱天干五行
      - day_zhi：日柱地支五行
      - majority：六柱综合众数
    - **hour_zhi**: 可选时辰地支（子/丑/寅/卯/辰/巳/午/未/申/酉/戌/亥），传入则输出时辰分析
    """
    try:
        parts = date.split('-')
        from datetime import date as date_cls
        solar_date = date_cls(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        raise HTTPException(status_code=400, detail=f"日期格式错误，请使用YYYY-MM-DD格式，如2026-05-22")

    if mode not in ("day_gan", "day_zhi", "majority"):
        raise HTTPException(status_code=400, detail=f"mode参数错误，可选：day_gan / day_zhi / majority")

    valid_zhi = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
    if hour_zhi and hour_zhi not in valid_zhi:
        raise HTTPException(status_code=400, detail=f"hour_zhi参数错误，可选：子/丑/寅/卯/辰/巳/午/未/申/酉/戌/亥")

    try:
        from lunarcalendar import Converter, Solar
        solar = Solar(solar_date.year, solar_date.month, solar_date.day)
        lunar = Converter.Solar2Lunar(solar)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"阴历转换失败: {str(e)}")

    # 计算干支
    _TIANGAN_LIST = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
    _DIZHI_LIST = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']

    # 年柱
    y_offset = solar_date.year - 1984
    year_gan = _TIANGAN_LIST[y_offset % 10]
    year_zhi = _DIZHI_LIST[y_offset % 12]

    # 日柱
    base_date = datetime(2000, 1, 7).date() if hasattr(datetime(2000, 1, 7), 'date') else __import__('datetime').date(2000, 1, 7)
    diff = (solar_date - base_date).days
    day_gan = _TIANGAN_LIST[diff % 10]
    day_zhi = _DIZHI_LIST[diff % 12]

    # 月柱
    month_dz_map = {1:'丑', 2:'寅', 3:'卯', 4:'辰', 5:'巳', 6:'午',
                    7:'未', 8:'申', 9:'酉', 10:'戌', 11:'亥', 12:'子'}
    month_zhi = month_dz_map[solar_date.month]
    tg_start_map = {'甲':'丙','己':'丙','乙':'戊','庚':'戊','丙':'庚','辛':'庚',
                     '丁':'壬','壬':'壬','戊':'甲','癸':'甲'}
    start_tg = tg_start_map[year_gan]
    start_idx = _TIANGAN_LIST.index(start_tg)
    month_dz_order = ['寅','卯','辰','巳','午','未','申','酉','戌','亥','子','丑']
    month_dz_idx = month_dz_order.index(month_zhi)
    month_gan = _TIANGAN_LIST[(start_idx + month_dz_idx) % 10]

    lunar_day = lunar.day

    # 六柱五行统计
    from collections import Counter
    six_pillars_wx = [
        _TIANGAN_MAP[year_gan]["wuxing"], _DIZHI_RED_MAP[year_zhi]["wuxing"],
        _TIANGAN_MAP[month_gan]["wuxing"], _DIZHI_RED_MAP[month_zhi]["wuxing"],
        _TIANGAN_MAP[day_gan]["wuxing"], _DIZHI_RED_MAP[day_zhi]["wuxing"],
    ]
    wx_counter = Counter(six_pillars_wx)

    # P1: 旺行判定逻辑
    if mode == "day_zhi":
        day_wuxing = _DIZHI_RED_MAP[day_zhi]["wuxing"]
        mode_desc = f"日柱地支{day_zhi}（{day_wuxing}行）"
    elif mode == "majority":
        day_wuxing = wx_counter.most_common(1)[0][0]
        mode_desc = f"六柱综合众数（{day_wuxing}行出现{wx_counter[day_wuxing]}次）"
    else:  # day_gan
        day_wuxing = _TIANGAN_MAP[day_gan]["wuxing"]
        mode_desc = f"日柱天干{day_gan}（{day_wuxing}行）"

    shengke = _get_shengke_info(day_wuxing)

    # 月相
    moon_phase = _MOON_PHASE.get(lunar_day, "未知")

    # 旺行号码
    wang_red = _fmt(_WUXING_MAP[shengke["旺行"]]["red_balls"])
    sheng_wo_red = _fmt(_WUXING_MAP[shengke["生我行"]]["red_balls"])
    wo_sheng_red = _fmt(_WUXING_MAP[shengke["我生行(泄)"]]["red_balls"])
    ke_wo_red = _fmt(_WUXING_MAP[shengke["克我行"]]["red_balls"])
    wo_ke_red = _fmt(_WUXING_MAP[shengke["我克行"]]["red_balls"])

    # 旺行蓝球
    wang_blue = _fmt(_WUXING_MAP[shengke["旺行"]]["blue_balls"])
    sheng_wo_blue = _fmt(_WUXING_MAP[shengke["生我行"]]["blue_balls"])
    ke_wo_blue = _fmt(_WUXING_MAP[shengke["克我行"]]["blue_balls"])

    # ===== P0: 月相号码 =====
    moon_red = _fmt(_MOON_PHASE_RED.get(moon_phase, [6, 16, 26]))
    moon_blue = _fmt(_MOON_PHASE_BLUE.get(moon_phase, [1]))
    moon_luck = _MOON_PHASE_LUCK.get(moon_phase, "")

    # ===== v3.0 P0-1: 六柱干支直接号码映射 =====
    liuzhu_parts = []
    liuzhu_red_all = []  # 收集六柱所有红球用于热度
    liuzhu_blue_all = []  # 收集六柱所有蓝球用于热度
    for pillar_label, tg, dz in [
        ("年柱", year_gan, year_zhi),
        ("月柱", month_gan, month_zhi),
        ("日柱", day_gan, day_zhi),
    ]:
        tg_red = _fmt(_TIANGAN_MAP[tg]["red_balls"])
        dz_red = _fmt(_DIZHI_RED_MAP[dz]["red_balls"])
        dz_blue = f"{_DIZHI_BLUE_MAP[dz]:02d}"
        liuzhu_parts.append(
            f"- {pillar_label} {tg}{dz}：{tg}→红球 {tg_red} ｜{dz}→红球 {dz_red} ｜蓝球 {dz_blue}"
        )
        liuzhu_red_all.extend(_TIANGAN_MAP[tg]["red_balls"])
        liuzhu_red_all.extend(_DIZHI_RED_MAP[dz]["red_balls"])
        liuzhu_blue_all.append(_DIZHI_BLUE_MAP[dz])

    formatted_liuzhu = (
        f"【六柱干支号码映射（娱乐）】\n"
        + "\n".join(liuzhu_parts)
    )

    # ===== v3.0 P0-2: 九宫飞星号码 =====
    year_star = _get_year_feixing(solar_date.year)
    # 飞星入中宫后，按洛书顺飞路径排列
    # 宫位顺序：5(中宫)→6(乾)→7(兑)→8(艮)→9(离)→1(坎)→2(坤)→3(震)→4(巽)
    _PALACE_ORDER = [5, 6, 7, 8, 9, 1, 2, 3, 4]
    feixing_parts = []
    feixing_red_all = []
    feixing_blue_all = []
    for i, palace in enumerate(_PALACE_ORDER):
        flying_star = (year_star - 1 + i) % 9 + 1  # 从年飞星开始递增
        palace_info = _JIUGONG_MAP[palace]
        palace_name = palace_info["name"]
        star_name = _FEIXING_NAMES.get(flying_star, f"{flying_star}")
        red_str = _fmt(palace_info["red_balls"])
        blue_str = _fmt(palace_info["blue_balls"])
        feixing_parts.append(f"  {star_name}→{palace_name}：红球 {red_str} ｜蓝球 {blue_str}")
        feixing_red_all.extend(palace_info["red_balls"])
        feixing_blue_all.extend(palace_info["blue_balls"])

    # 日支对应八卦方位
    day_zhi_bagua = _DIZHI_BAGUA_MAP[day_zhi]

    formatted_feixing = (
        f"【九宫飞星号码（娱乐）】\n"
        f"年飞星{year_star}入中宫，九宫飞星排列：\n"
        + "\n".join(feixing_parts) + "\n"
        f"\n日支{day_zhi}→{day_zhi_bagua['gua']}卦·{day_zhi_bagua['fangwei']}方："
        f"红球 {_fmt(day_zhi_bagua['red_balls'])} ｜蓝球 {_fmt(day_zhi_bagua['blue_balls'])}"
    )

    # ===== v3.0 P1: 纳音五行号码 =====
    day_ganzhi = day_gan + day_zhi
    day_nayin = _NAYIN_MAP.get(day_ganzhi, "")
    nayin_wuxing = _NAYIN_WUXING.get(day_nayin, "")
    nayin_wuxing_label = f"{day_nayin}（{nayin_wuxing}行）" if day_nayin else "未知"
    # 纳音与正五行冲突判定
    nayin_conflict = False
    if nayin_wuxing and nayin_wuxing != day_wuxing:
        nayin_conflict = True
    nayin_red = _fmt(_WUXING_MAP[nayin_wuxing]["red_balls"]) if nayin_wuxing else "无"
    nayin_blue = _fmt(_WUXING_MAP[nayin_wuxing]["blue_balls"]) if nayin_wuxing else "无"
    nayin_conflict_mark = " ⚠️与正五行冲突" if nayin_conflict else ""

    # ===== v3.0 P2: 时辰号码分析（可选） =====
    hour_parts = []
    hour_red_all = []
    hour_blue_all = []
    if hour_zhi:
        hour_wuxing = _DIZHI_RED_MAP[hour_zhi]["wuxing"]
        hour_red = _fmt(_DIZHI_RED_MAP[hour_zhi]["red_balls"])
        hour_blue = f"{_DIZHI_BLUE_MAP[hour_zhi]:02d}"
        hour_wuxing_red = _fmt(_WUXING_MAP[hour_wuxing]["red_balls"])
        hour_wuxing_blue = _fmt(_WUXING_MAP[hour_wuxing]["blue_balls"])
        hour_luck = _HOUR_LUCK.get(hour_zhi, "")
        hour_parts.append(
            f"【时辰号码分析（娱乐）】\n"
            f"- 时辰：{hour_zhi}时（{hour_wuxing}行）\n"
            f"- 时支号码：红球 {hour_red} ｜蓝球 {hour_blue}\n"
            f"- 时辰五行号码：红球 {hour_wuxing_red} ｜蓝球 {hour_wuxing_blue}\n"
            f"- 时辰提示：{hour_luck}"
        )
        hour_red_all.extend(_DIZHI_RED_MAP[hour_zhi]["red_balls"])
        hour_red_all.extend(_WUXING_MAP[hour_wuxing]["red_balls"])
        hour_blue_all.append(_DIZHI_BLUE_MAP[hour_zhi])
        hour_blue_all.extend(_WUXING_MAP[hour_wuxing]["blue_balls"])

    # ===== v3.0 P3: 号码热度汇总（升级版） =====
    red_heat = {}
    blue_heat = {}

    # 维度1：旺行（权重×2）
    for n in _WUXING_MAP[shengke["旺行"]]["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 2
    for n in _WUXING_MAP[shengke["旺行"]]["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 2

    # 维度2：生我行（权重×1）
    for n in _WUXING_MAP[shengke["生我行"]]["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in _WUXING_MAP[shengke["生我行"]]["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度3：日月
    for n in _SUN_MOON_MAP["日"]["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in _SUN_MOON_MAP["月"]["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in _SUN_MOON_MAP["日"]["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 1
    for n in _SUN_MOON_MAP["月"]["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度4：月相
    for n in _MOON_PHASE_RED.get(moon_phase, []):
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in _MOON_PHASE_BLUE.get(moon_phase, []):
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度5：六柱干支（权重×1）
    for n in liuzhu_red_all:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in liuzhu_blue_all:
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度6：纳音五行（权重×1）
    if nayin_wuxing:
        for n in _WUXING_MAP[nayin_wuxing]["red_balls"]:
            red_heat[n] = red_heat.get(n, 0) + 1
        for n in _WUXING_MAP[nayin_wuxing]["blue_balls"]:
            blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度7：九宫飞星当日宫位号码（权重×1）
    for n in day_zhi_bagua["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in day_zhi_bagua["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度8：时辰（可选，权重×1）
    if hour_zhi:
        for n in hour_red_all:
            red_heat[n] = red_heat.get(n, 0) + 1
        for n in hour_blue_all:
            blue_heat[n] = blue_heat.get(n, 0) + 1

    # 按热度排序
    sorted_red = sorted(red_heat.items(), key=lambda x: (-x[1], x[0]))
    sorted_blue = sorted(blue_heat.items(), key=lambda x: (-x[1], x[0]))

    def _heat_label(count):
        if count >= 5: return "⭐⭐⭐⭐"
        elif count >= 3: return "⭐⭐⭐"
        elif count == 2: return "⭐⭐"
        else: return "⭐"

    red_summary_parts = []
    for num, cnt in sorted_red[:15]:  # v3.0: 取top15红球
        red_summary_parts.append(f"{num:02d}{_heat_label(cnt)}")

    blue_summary_parts = []
    for num, cnt in sorted_blue[:8]:  # v3.0: 取top8蓝球
        blue_summary_parts.append(f"{num:02d}{_heat_label(cnt)}")

    # 三星号码（3+次重合）
    three_star_red = [f"{n:02d}" for n, c in sorted_red if c >= 3]
    three_star_blue = [f"{n:02d}" for n, c in sorted_blue if c >= 3]

    # 冷门号码（热度为0）
    all_red_nums = set(range(1, 34))
    all_blue_nums = set(range(1, 17))
    hot_red = set(red_heat.keys())
    hot_blue = set(blue_heat.keys())
    cold_red = sorted(all_red_nums - hot_red)
    cold_blue = sorted(all_blue_nums - hot_blue)

    # 冲突号码（纳音与正五行冲突时的号码）
    conflict_red = []
    conflict_blue = []
    if nayin_conflict and nayin_wuxing:
        nayin_red_set = set(_WUXING_MAP[nayin_wuxing]["red_balls"])
        nayin_blue_set = set(_WUXING_MAP[nayin_wuxing]["blue_balls"])
        zheng_red_set = set(_WUXING_MAP[day_wuxing]["red_balls"])
        zheng_blue_set = set(_WUXING_MAP[day_wuxing]["blue_balls"])
        conflict_red = sorted(nayin_red_set - zheng_red_set)  # 纳音有但正五行没有的红球
        conflict_blue = sorted(nayin_blue_set - zheng_blue_set)

    # ===== 生成6个预格式化文本 =====
    # formatted_shengke（升级：+纳音行+冲突标记）
    shengke_lines = [
        f"【五行生克分析（娱乐）】",
        f"基于{mode_desc}的五行生克关系：",
        f"- 旺行（{shengke['旺行']}）：红球 {wang_red} ｜蓝球 {wang_blue}",
        f"- 生我行（{shengke['生我行']}→{shengke['旺行']}）：红球 {sheng_wo_red} ｜蓝球 {sheng_wo_blue}",
        f"- 我生行·泄（{shengke['旺行']}→{shengke['我生行(泄)']}）：{wo_sheng_red}",
        f"- 克我行（{shengke['克我行']}→{shengke['旺行']}）：红球 {ke_wo_red} ｜蓝球 {ke_wo_blue}",
        f"- 我克行（{shengke['旺行']}→{shengke['我克行']}）：{wo_ke_red}",
        f"- 纳音五行（{day_ganzhi}→{nayin_wuxing_label}）：红球 {nayin_red} ｜蓝球 {nayin_blue}{nayin_conflict_mark}",
    ]
    formatted_shengke = "\n".join(shengke_lines)

    formatted_sun_moon = (
        f"【日月水火分析（娱乐）】\n"
        f"- 日·太阳（{_SUN_MOON_MAP['日']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['日']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['日']['blue_balls'])}\n"
        f"- 月·太阴（{_SUN_MOON_MAP['月']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['月']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['月']['blue_balls'])}\n"
        f"- 水·坎卦（{_SUN_MOON_MAP['水']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['水']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['水']['blue_balls'])}\n"
        f"- 火·离卦（{_SUN_MOON_MAP['火']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['火']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['火']['blue_balls'])}"
    )

    formatted_moon_phase = (
        f"【月相分析（娱乐）】\n"
        f"- 今日阴历日数：{lunar_day}\n"
        f"- 今日月相：{moon_phase}\n"
        f"- 月相红球：{moon_red}\n"
        f"- 月相蓝球：{moon_blue}\n"
        f"- 月相提示：{moon_luck}"
    )

    # formatted_summary（升级版：+新维度+冲突/冷门号）
    three_star_red_str = "、".join(three_star_red) if three_star_red else "无"
    three_star_blue_str = "、".join(three_star_blue) if three_star_blue else "无"
    cold_red_str = "、".join(f"{n:02d}" for n in cold_red) if cold_red else "无"
    cold_blue_str = "、".join(f"{n:02d}" for n in cold_blue) if cold_blue else "无"
    conflict_red_str = "、".join(f"{n:02d}" for n in conflict_red) if conflict_red else "无"
    conflict_blue_str = "、".join(f"{n:02d}" for n in conflict_blue) if conflict_blue else "无"

    dimension_count = 8 if hour_zhi else 7
    summary_lines = [
        f"【综合号码热度汇总（娱乐）】",
        f"以下号码在{dimension_count}个维度（旺行+生我行+日月+月相+六柱干支+纳音五行+飞星方位{'+时辰' if hour_zhi else ''}）重合出现，⭐越多重合度越高：",
        f"",
        f"🔥 红球热度TOP15：{'  '.join(red_summary_parts)}",
        f"🔵 蓝球热度TOP8：{'  '.join(blue_summary_parts)}",
        f"",
        f"⭐⭐⭐ 三星红球（3+维度重合）：{three_star_red_str}",
        f"⭐⭐⭐ 三星蓝球（3+维度重合）：{three_star_blue_str}",
    ]
    if nayin_conflict:
        summary_lines.extend([
            f"",
            f"⚠️ 纳音冲突红球（纳音有·正五行无）：{conflict_red_str}",
            f"⚠️ 纳音冲突蓝球（纳音有·正五行无）：{conflict_blue_str}",
        ])
    summary_lines.extend([
        f"",
        f"❄️ 冷门红球（无维度覆盖）：{cold_red_str}",
        f"❄️ 冷门蓝球（无维度覆盖）：{cold_blue_str}",
        f"",
        f"💡 旺行判定模式：{mode_desc}",
    ])
    formatted_summary = "\n".join(summary_lines)

    result = {
        "formatted_shengke": formatted_shengke,
        "formatted_sun_moon": formatted_sun_moon,
        "formatted_moon_phase": formatted_moon_phase,
        "formatted_summary": formatted_summary,
        "formatted_liuzhu": formatted_liuzhu,
        "formatted_feixing": formatted_feixing,
    }

    # 时辰号码（可选，拼入summary末尾）
    if hour_parts:
        result["formatted_hour"] = "\n".join(hour_parts)

    return result


@app.get("/ganzhi/map", tags=["双色球玄学映射"])
async def ganzhi_map(
    year_gan: str = "甲", year_zhi: str = "子",
    month_gan: str = "甲", month_zhi: str = "子",
    day_gan: str = "甲", day_zhi: str = "子",
    lunar_day: int = 1,
):
    """
    双色球玄学号码映射接口 - 根据天干地支返回对应红球蓝球号码。
    参数从知识库查询结果中获取，传入本接口获取准确的号码映射。
    """
    # 天干映射
    yg = _TIANGAN_MAP.get(year_gan)
    mg = _TIANGAN_MAP.get(month_gan)
    dg = _TIANGAN_MAP.get(day_gan)
    # 地支映射
    yz = _DIZHI_RED_MAP.get(year_zhi)
    mz = _DIZHI_RED_MAP.get(month_zhi)
    dz = _DIZHI_RED_MAP.get(day_zhi)

    if not all([yg, mg, dg, yz, mz, dz]):
        raise HTTPException(status_code=400, detail=f"无效的天干或地支参数。天干可选：甲乙丙丁戊己庚辛壬癸；地支可选：子丑寅卯辰巳午未申酉戌亥")

    # 地支蓝球
    yz_blue = _DIZHI_BLUE_MAP.get(year_zhi, 0)
    mz_blue = _DIZHI_BLUE_MAP.get(month_zhi, 0)
    dz_blue = _DIZHI_BLUE_MAP.get(day_zhi, 0)

    # 天干蓝球（按五行取蓝球）
    _wuxing_blue_map = {"金": _WUXING_MAP["金"]["blue_balls"], "木": _WUXING_MAP["木"]["blue_balls"], "水": _WUXING_MAP["水"]["blue_balls"], "火": _WUXING_MAP["火"]["blue_balls"], "土": _WUXING_MAP["土"]["blue_balls"]}
    yg_blue = _fmt(_wuxing_blue_map[yg["wuxing"]])
    mg_blue = _fmt(_wuxing_blue_map[mg["wuxing"]])
    dg_blue = _fmt(_wuxing_blue_map[dg["wuxing"]])

    # 五行生克分析（基于日柱天干五行）
    day_wuxing = dg["wuxing"]
    shengke = _get_shengke_info(day_wuxing)

    # 月相
    moon_phase = _MOON_PHASE.get(lunar_day, "未知")

    # 旺行/生行/克行的红球号码（仅用于formatted_shengke，不再单独返回）
    wang_red = _fmt(_WUXING_MAP[shengke["旺行"]]["red_balls"])
    sheng_wo_red = _fmt(_WUXING_MAP[shengke["生我行"]]["red_balls"])
    wo_sheng_red = _fmt(_WUXING_MAP[shengke["我生行(泄)"]]["red_balls"])
    ke_wo_red = _fmt(_WUXING_MAP[shengke["克我行"]]["red_balls"])
    wo_ke_red = _fmt(_WUXING_MAP[shengke["我克行"]]["red_balls"])

    result = {
        # ===== 年柱映射 =====
        "year_gan_name": year_gan,
        "year_gan_wuxing": yg["wuxing"],
        "year_gan_red_balls": _fmt(yg["red_balls"]),
        "year_gan_blue_balls": yg_blue,
        "year_zhi_name": year_zhi,
        "year_zhi_wuxing": yz["wuxing"],
        "year_zhi_red_balls": _fmt(yz["red_balls"]),
        "year_zhi_blue_ball": f"{yz_blue:02d}",

        # ===== 月柱映射 =====
        "month_gan_name": month_gan,
        "month_gan_wuxing": mg["wuxing"],
        "month_gan_red_balls": _fmt(mg["red_balls"]),
        "month_gan_blue_balls": mg_blue,
        "month_zhi_name": month_zhi,
        "month_zhi_wuxing": mz["wuxing"],
        "month_zhi_red_balls": _fmt(mz["red_balls"]),
        "month_zhi_blue_ball": f"{mz_blue:02d}",

        # ===== 日柱映射 =====
        "day_gan_name": day_gan,
        "day_gan_wuxing": dg["wuxing"],
        "day_gan_red_balls": _fmt(dg["red_balls"]),
        "day_gan_blue_balls": dg_blue,
        "day_zhi_name": day_zhi,
        "day_zhi_wuxing": dz["wuxing"],
        "day_zhi_red_balls": _fmt(dz["red_balls"]),
        "day_zhi_blue_ball": f"{dz_blue:02d}",

        # ===== 五行号码总表 =====
        "wuxing_jin_red": _fmt(_WUXING_MAP["金"]["red_balls"]),
        "wuxing_jin_blue": _fmt(_WUXING_MAP["金"]["blue_balls"]),
        "wuxing_mu_red": _fmt(_WUXING_MAP["木"]["red_balls"]),
        "wuxing_mu_blue": _fmt(_WUXING_MAP["木"]["blue_balls"]),
        "wuxing_shui_red": _fmt(_WUXING_MAP["水"]["red_balls"]),
        "wuxing_shui_blue": _fmt(_WUXING_MAP["水"]["blue_balls"]),
        "wuxing_huo_red": _fmt(_WUXING_MAP["火"]["red_balls"]),
        "wuxing_huo_blue": _fmt(_WUXING_MAP["火"]["blue_balls"]),
        "wuxing_tu_red": _fmt(_WUXING_MAP["土"]["red_balls"]),
        "wuxing_tu_blue": _fmt(_WUXING_MAP["土"]["blue_balls"]),

        # ===== 预格式化输出（Agent直接复制粘贴，唯一数据源） =====
        "formatted_shengke": (
            f"【五行生克分析（娱乐）】\n"
            f"基于日柱天干{day_gan}（{day_wuxing}行）的五行生克关系：\n"
            f"- 旺行（{shengke['旺行']}）：{wang_red}\n"
            f"- 生我行（{shengke['生我行']}→{shengke['旺行']}）：{sheng_wo_red}\n"
            f"- 我生行·泄（{shengke['旺行']}→{shengke['我生行(泄)']}）：{wo_sheng_red}\n"
            f"- 克我行（{shengke['克我行']}→{shengke['旺行']}）：{ke_wo_red}\n"
            f"- 我克行（{shengke['旺行']}→{shengke['我克行']}）：{wo_ke_red}"
        ),
        "formatted_sun_moon": (
            f"【日月水火分析（娱乐）】\n"
            f"- 日·太阳（{_SUN_MOON_MAP['日']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['日']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['日']['blue_balls'])}\n"
            f"- 月·太阴（{_SUN_MOON_MAP['月']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['月']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['月']['blue_balls'])}\n"
            f"- 水·坎卦（{_SUN_MOON_MAP['水']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['水']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['水']['blue_balls'])}\n"
            f"- 火·离卦（{_SUN_MOON_MAP['火']['desc']}）：红球 {_fmt(_SUN_MOON_MAP['火']['red_balls'])} ｜蓝球 {_fmt(_SUN_MOON_MAP['火']['blue_balls'])}"
        ),
        "formatted_moon_phase": (
            f"【月相分析（娱乐）】\n"
            f"- 今日阴历日数：{lunar_day}\n"
            f"- 今日月相：{moon_phase}"
        ),
    }

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
