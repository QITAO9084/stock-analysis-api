from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import json
import sys
import os as _os

# V5.20.24: ADX过滤+MACD柱限权+信号方向覆盖（V5.20.x修复重新应用）
print(f"===== MODULE LOADED: sys.argv={sys.argv}, PORT={_os.environ.get('PORT', 'NOT SET')}, RAILWAY_ENV={_os.environ.get('RAILWAY_ENVIRONMENT', 'NOT SET')} =====", flush=True)
import threading

app = FastAPI(
    title="Stock Analysis API",
    description="股票/加密货币分析API - V5（含买卖点检测、缓存重试限速）",
    version="5.2.0"
)

# Coze兼容：/openapi.json/xxx → /xxx 路径重写
class CozePathRewriteMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/openapi.json/"):
            new_path = path[len("/openapi.json"):]  # 去掉 /openapi.json 前缀
            # 构造新的URL scope
            request.scope["path"] = new_path
            request.scope["raw_path"] = new_path.encode()
        response = await call_next(request)
        return response

app.add_middleware(CozePathRewriteMiddleware)

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

# V5.20.20: ADX 趋势强度计算
def calculate_adx(data, period=14):
    """计算 ADX 趋势强度指标（Wilder's DMI）"""
    if len(data) < period * 2:
        return {"adx": 0, "plus_di": 0, "minus_di": 0, "trend": "ranging"}

    high = data['High']
    low = data['Low']
    close = data['Close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    # +DM and -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    adx_val = round(float(adx.iloc[-1]), 2)
    plus_val = round(float(plus_di.iloc[-1]), 2)
    minus_val = round(float(minus_di.iloc[-1]), 2)

    if adx_val > 25:
        trend = "strong_bull" if plus_val > minus_val else "strong_bear"
    elif adx_val > 20:
        trend = "weak_bull" if plus_val > minus_val else "weak_bear"
    else:
        trend = "ranging"

    return {"adx": adx_val, "plus_di": plus_val, "minus_di": minus_val, "trend": trend}

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

    # 2. MACD评分（含交叉检测 + V5.20.20 柱强度限权）
    # 柱强度：MACD柱占价格百分比，避免 0.16% 柱高给满分
    _macd_hist_pct = abs(macd_data['histogram']) / current_price * 100 if current_price > 0 else 0
    if _macd_hist_pct >= 0.3:
        _hist_strength = "full"   # ≥0.3% → 满分
    elif _macd_hist_pct >= 0.1:
        _hist_strength = "medium"  # 0.1-0.3% → 中档
    else:
        _hist_strength = "min"    # <0.1% → 最低分

    if macd_data['golden_cross']:
        buy_signals.append("MACD今日金叉，强烈买入信号")
        buy_score += 3  # 金叉事件不降权
    elif macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
        _pts = 2 if _hist_strength == "full" else (1 if _hist_strength == "medium" else 0.5)
        buy_signals.append(f"MACD多头运行（柱强{_macd_hist_pct:.2f}%），趋势偏多")
        buy_score += _pts

    if macd_data['death_cross']:
        sell_signals.append("MACD今日死叉，强烈卖出信号")
        sell_score += 3  # 死叉事件不降权
    elif macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
        _pts = 2 if _hist_strength == "full" else (1 if _hist_strength == "medium" else 0.5)
        sell_signals.append(f"MACD空头运行（柱强{_macd_hist_pct:.2f}%），趋势偏空")
        sell_score += _pts

    # MACD柱状图趋势：柱子缩小 = 动能衰减
    if macd_data['histogram_trend'] == "shrinking" and abs(macd_data['histogram']) > 0.5:
        _pts = 1 if _hist_strength == "full" else (0.5 if _hist_strength == "medium" else 0)
        if macd_data['histogram'] > 0:
            sell_signals.append(f"MACD多头柱缩小（柱强{_macd_hist_pct:.2f}%），上涨动能衰减")
            sell_score += _pts
        else:
            buy_signals.append(f"MACD空头柱缩小（柱强{_macd_hist_pct:.2f}%），下跌动能衰减")
            buy_score += _pts

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

@app.get("/health")
def health_check():
    """Railway 健康检查端点"""
    return {"status": "healthy"}

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

        # V5.20.24: ADX 过滤 + 信号方向覆盖
        adx_data = calculate_adx(data)
        adx_val = adx_data["adx"]
        adx_trend = adx_data["trend"]

        # 最终信号（默认用原始值，ADX会覆盖）
        final_signal = signal_data["signal"]
        final_confidence = signal_data["confidence"]
        final_trade_point = trade_points["trade_point"]
        final_trade_point_cn = trade_point_cn.get(trade_points["trade_point"], "观望")

        if adx_val < 25:
            # 震荡市：强制中性，禁止买卖信号泄露
            final_signal = "NEUTRAL"
            final_confidence = "LOW"
            final_trade_point = "hold"
            final_trade_point_cn = f"ADX震荡过滤-观望（ADX={adx_val}）"
        else:
            # ADX≥25 强趋势：防止 event-driven 评分为 hold 但加权方向明确
            if trade_points["trade_point"] == "hold":
                orig_signal = signal_data["signal"]
                if orig_signal == "BUY":
                    final_trade_point = "buy"
                    final_trade_point_cn = f"建议买入（ADX={adx_val}，{adx_trend}）"
                elif orig_signal == "SELL":
                    final_trade_point = "sell"
                    final_trade_point_cn = f"建议卖出（ADX={adx_val}，{adx_trend}）"

        return {
            # 基础信息
            "symbol": str(symbol),
            "name": str(info.get("longName", "N/A")),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": str(info.get("currency", "USD")),
            "market": str(market),
            "analysis_time": datetime.now().isoformat(),
            # 买卖信号（V5.20.24: 使用 ADX 过滤后的 final_* 值）
            "signal": str(final_signal),
            "confidence": str(final_confidence),
            "key_signals_text": signals_text,
            # 买卖点（V5.20.24: 使用 ADX 过滤后的值）
            "trade_point": str(final_trade_point),
            "trade_point_cn": final_trade_point_cn,
            "trade_score": trade_points["score"],
            # ADX 过滤（V5.20.24 新增）
            "adx": adx_val,
            "adx_trend": adx_trend,
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

# ===== P16: 二十八宿映射表（宇宙维度）=====
# 28宿按4象7宿排列，每宿对应五行属性
# 值宿计算：儒略日 JD mod 28 → 宿索引
_28XIU_MAP = [
    # 东方青龙7宿
    {"name": "角", "xiang": "青龙", "wuxing": "木", "animal": "蛟", "desc": "角木蛟"},
    {"name": "亢", "xiang": "青龙", "wuxing": "金", "animal": "龙", "desc": "亢金龙"},
    {"name": "氐", "xiang": "青龙", "wuxing": "土", "animal": "貉", "desc": "氐土貉"},
    {"name": "房", "xiang": "青龙", "wuxing": "火", "animal": "兔", "desc": "房日兔"},
    {"name": "心", "xiang": "青龙", "wuxing": "火", "animal": "狐", "desc": "心月狐"},
    {"name": "尾", "xiang": "青龙", "wuxing": "火", "animal": "虎", "desc": "尾火虎"},
    {"name": "箕", "xiang": "青龙", "wuxing": "水", "animal": "豹", "desc": "箕水豹"},
    # 北方玄武7宿
    {"name": "斗", "xiang": "玄武", "wuxing": "木", "animal": "獬", "desc": "斗木獬"},
    {"name": "牛", "xiang": "玄武", "wuxing": "金", "animal": "牛", "desc": "牛金牛"},
    {"name": "女", "xiang": "玄武", "wuxing": "土", "animal": "蝠", "desc": "女士蝠"},
    {"name": "虚", "xiang": "玄武", "wuxing": "火", "animal": "鼠", "desc": "虚日鼠"},
    {"name": "危", "xiang": "玄武", "wuxing": "火", "animal": "燕", "desc": "危月燕"},
    {"name": "室", "xiang": "玄武", "wuxing": "火", "animal": "猪", "desc": "室火猪"},
    {"name": "壁", "xiang": "玄武", "wuxing": "水", "animal": "貐", "desc": "壁水貐"},
    # 西方白虎7宿
    {"name": "奎", "xiang": "白虎", "wuxing": "木", "animal": "狼", "desc": "奎木狼"},
    {"name": "娄", "xiang": "白虎", "wuxing": "金", "animal": "狗", "desc": "娄金狗"},
    {"name": "胃", "xiang": "白虎", "wuxing": "土", "animal": "雉", "desc": "胃土雉"},
    {"name": "昴", "xiang": "白虎", "wuxing": "火", "animal": "鸡", "desc": "昴日鸡"},
    {"name": "毕", "xiang": "白虎", "wuxing": "火", "animal": "乌", "desc": "毕月乌"},
    {"name": "觜", "xiang": "白虎", "wuxing": "火", "animal": "猴", "desc": "觜火猴"},
    {"name": "参", "xiang": "白虎", "wuxing": "水", "animal": "猿", "desc": "参水猿"},
    # 南方朱雀7宿
    {"name": "井", "xiang": "朱雀", "wuxing": "木", "animal": "犴", "desc": "井木犴"},
    {"name": "鬼", "xiang": "朱雀", "wuxing": "金", "animal": "羊", "desc": "鬼金羊"},
    {"name": "柳", "xiang": "朱雀", "wuxing": "土", "animal": "獐", "desc": "柳土獐"},
    {"name": "星", "xiang": "朱雀", "wuxing": "火", "animal": "马", "desc": "星日马"},
    {"name": "张", "xiang": "朱雀", "wuxing": "火", "animal": "鹿", "desc": "张月鹿"},
    {"name": "翼", "xiang": "朱雀", "wuxing": "火", "animal": "蛇", "desc": "翼火蛇"},
    {"name": "轸", "xiang": "朱雀", "wuxing": "水", "animal": "蚓", "desc": "轸水蚓"},
]

# 七曜对应五行（日月+五大行星）
_QIYAO_MAP = {
    "日": {"wuxing": "火", "desc": "太阳·火行"},
    "月": {"wuxing": "水", "desc": "太阴·水行"},
    "火": {"wuxing": "火", "desc": "荧惑·火行"},
    "水": {"wuxing": "水", "desc": "辰星·水行"},
    "木": {"wuxing": "木", "desc": "岁星·木行"},
    "金": {"wuxing": "金", "desc": "太白·金行"},
    "土": {"wuxing": "土", "desc": "镇星·土行"},
}

# 七曜星期映射（星期日=日，星期一=月，...）
_WEEKDAY_QIYAO = {
    6: "日",  # 周日→日
    0: "月",  # 周一→月
    1: "火",  # 周二→火（Mars→Tuesday）
    2: "水",  # 周三→水（Mercury→Wednesday）
    3: "木",  # 周四→木（Jupiter→Thursday）
    4: "金",  # 周五→金（Venus→Friday）
    5: "土",  # 周六→土（Saturn→Saturday）
}


def _get_zhixiu(solar_date):
    """
    计算当日值宿（二十八宿）
    使用儒略日对28取模
    基准：2000-01-07 = 角宿（JD=2451551, 2451551 mod 28 = 7）
    """
    # 简化儒略日计算
    y = solar_date.year
    m = solar_date.month
    d = solar_date.day
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5
    idx = int(jd) % 28
    return _28XIU_MAP[idx]


def _get_qiyao(solar_date):
    """
    计算当日七曜（日月+五星）
    星期映射：日(周日) 月(周一) 火(周二) 水(周三) 木(周四) 金(周五) 土(周六)
    """
    weekday = solar_date.weekday()  # 0=Mon, 6=Sun
    yao_name = _WEEKDAY_QIYAO[weekday]
    return _QIYAO_MAP[yao_name]


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
async def ganzhi_by_date(date: str = "2026-05-22", mode: str = "day_gan", hour_zhi: str = "", birthday: str = ""):
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

    if mode not in ("day_gan", "day_zhi", "majority", "auto"):
        raise HTTPException(status_code=400, detail=f"mode参数错误，可选：day_gan / day_zhi / majority / auto")

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

    # v3.4: birthday参数计算（base_date和干支列表已定义）
    b_day_gan = b_day_zhi = b_day_wuxing = None
    b_shengke = None
    if birthday:
        try:
            b_parts = birthday.split('-')
            b_date = date_cls(int(b_parts[0]), int(b_parts[1]), int(b_parts[2]))
            b_diff = (b_date - base_date).days
            b_day_gan = _TIANGAN_LIST[b_diff % 10]
            b_day_zhi = _DIZHI_LIST[b_diff % 12]
            b_day_wuxing = _TIANGAN_MAP[b_day_gan]["wuxing"]
            b_shengke = _get_shengke_info(b_day_wuxing)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"birthday参数格式错误，请使用YYYY-MM-DD格式")

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
    auto_reason = ""
    if mode == "auto":
        # v3.5: auto模式自动推荐
        top_wx, top_count = wx_counter.most_common(1)[0]
        day_gan_wx = _TIANGAN_MAP[day_gan]["wuxing"]
        day_zhi_wx = _DIZHI_RED_MAP[day_zhi]["wuxing"]
        if top_count >= 4:
            mode = "majority"
            auto_reason = f"auto→majority：众数{top_wx}出现{top_count}次"
        elif day_zhi_wx != day_gan_wx and top_count >= 3:
            mode = "day_zhi"
            auto_reason = f"auto→day_zhi：众数{top_wx}出现{top_count}次，日干≠日支"
        else:
            mode = "day_gan"
            auto_reason = f"auto→day_gan：六柱分散，默认日干"

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

    # v3.4: 追加出生日柱到 formatted_liuzhu
    if birthday and b_day_gan and b_day_zhi:
        b_tg_red = _fmt(_TIANGAN_MAP[b_day_gan]["red_balls"])
        b_dz_red = _fmt(_DIZHI_RED_MAP[b_day_zhi]["red_balls"])
        b_dz_blue = f"{_DIZHI_BLUE_MAP[b_day_zhi]:02d}"
        birth_pillar = f"- 🎂出生日柱 {b_day_gan}{b_day_zhi}：{b_day_gan}→红球 {b_tg_red} ｜{b_day_zhi}→红球 {b_dz_red} ｜蓝球 {b_dz_blue}"
        formatted_liuzhu += "\n" + birth_pillar
        liuzhu_red_all.extend(_TIANGAN_MAP[b_day_gan]["red_balls"])
        liuzhu_red_all.extend(_DIZHI_RED_MAP[b_day_zhi]["red_balls"])
        liuzhu_blue_all.append(_DIZHI_BLUE_MAP[b_day_zhi])


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

    # ===== v5.0 P16: 二十八宿+七曜（宇宙维度）=====
    zhixiu = _get_zhixiu(solar_date)
    qiyao = _get_qiyao(solar_date)
    xiu_wuxing = zhixiu["wuxing"]
    yao_wuxing = qiyao["wuxing"]
    xiu_conflict = xiu_wuxing != day_wuxing
    yao_conflict = yao_wuxing != day_wuxing
    formatted_xingxiu = (
        f"【二十八宿·七曜号码（娱乐）】\n"
        f"值宿：{zhixiu['xiang']}{zhixiu['desc']}·{xiu_wuxing}行"
        f"{' ⚠️与正五行冲突' if xiu_conflict else ''}\n"
        f"  值宿红球 {_fmt(_WUXING_MAP[xiu_wuxing]['red_balls'])} ｜蓝球 {_fmt(_WUXING_MAP[xiu_wuxing]['blue_balls'])}\n"
        f"七曜：{qiyao['desc']}·{yao_wuxing}行"
        f"{' ⚠️与正五行冲突' if yao_conflict else ''}\n"
        f"  七曜红球 {_fmt(_WUXING_MAP[yao_wuxing]['red_balls'])} ｜蓝球 {_fmt(_WUXING_MAP[yao_wuxing]['blue_balls'])}\n"
        f"🌌三维天文坐标：月相{moon_phase} + 宿{zhixiu['name']}({xiu_wuxing}) + 曜{qiyao['desc'][:2]}({yao_wuxing})"
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

    # 维度7b：二十八宿值宿（权重×1，v5.0 P16宇宙维度）
    for n in _WUXING_MAP[xiu_wuxing]["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in _WUXING_MAP[xiu_wuxing]["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度7c：七曜照宫（权重×1，v5.0 P16宇宙维度）
    for n in _WUXING_MAP[yao_wuxing]["red_balls"]:
        red_heat[n] = red_heat.get(n, 0) + 1
    for n in _WUXING_MAP[yao_wuxing]["blue_balls"]:
        blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度8：时辰（可选，权重×1）
    if hour_zhi:
        for n in hour_red_all:
            red_heat[n] = red_heat.get(n, 0) + 1
        for n in hour_blue_all:
            blue_heat[n] = blue_heat.get(n, 0) + 1

    # 维度9：出生信息（v3.5，生我行权重×2，克我行×0负面）
    if birthday and b_shengke:
        for n in _WUXING_MAP[b_shengke["旺行"]]["red_balls"]:
            red_heat[n] = red_heat.get(n, 0) + 1
        for n in _WUXING_MAP[b_shengke["旺行"]]["blue_balls"]:
            blue_heat[n] = blue_heat.get(n, 0) + 1
        for n in _WUXING_MAP[b_shengke["生我行"]]["red_balls"]:
            red_heat[n] = red_heat.get(n, 0) + 2  # v3.5: +3.03%有效，权重×2
        for n in _WUXING_MAP[b_shengke["生我行"]]["blue_balls"]:
            blue_heat[n] = blue_heat.get(n, 0) + 1
        # v3.5: 克我行-3.25%负面，不参与热度
        for n in _TIANGAN_MAP[b_day_gan]["red_balls"]:
            red_heat[n] = red_heat.get(n, 0) + 1
        for n in _DIZHI_RED_MAP[b_day_zhi]["red_balls"]:
            red_heat[n] = red_heat.get(n, 0) + 1
        blue_heat[_DIZHI_BLUE_MAP[b_day_zhi]] = blue_heat.get(_DIZHI_BLUE_MAP[b_day_zhi], 0) + 1

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

    # v3.4: 出生维度追加到 formatted_shengke
    if birthday and b_shengke:
        b_wang_red = _fmt(_WUXING_MAP[b_shengke["旺行"]]["red_balls"])
        b_wang_blue = _fmt(_WUXING_MAP[b_shengke["旺行"]]["blue_balls"])
        b_sheng_wo_red = _fmt(_WUXING_MAP[b_shengke["生我行"]]["red_balls"])
        b_sheng_wo_blue = _fmt(_WUXING_MAP[b_shengke["生我行"]]["blue_balls"])
        b_wo_sheng_red = _fmt(_WUXING_MAP[b_shengke["我生行(泄)"]]["red_balls"])
        b_ke_wo_red = _fmt(_WUXING_MAP[b_shengke["克我行"]]["red_balls"])
        b_ke_wo_blue = _fmt(_WUXING_MAP[b_shengke["克我行"]]["blue_balls"])
        b_wo_ke_red = _fmt(_WUXING_MAP[b_shengke["我克行"]]["red_balls"])
        birth_lines = [
            "",
            f"🎂 出生维度（{birthday}·{b_day_gan}{b_day_zhi}日·{b_day_wuxing}行）：",
            f"- 出生旺行（{b_shengke['旺行']}）：红球 {b_wang_red} ｜蓝球 {b_wang_blue}",
            f"- 出生生我行（{b_shengke['生我行']}→{b_shengke['旺行']}）：红球 {b_sheng_wo_red} ｜蓝球 {b_sheng_wo_blue}",
            f"- 出生我生行·泄（{b_shengke['旺行']}→{b_shengke['我生行(泄)']}）：{b_wo_sheng_red}",
            f"- 出生克我行（{b_shengke['克我行']}→{b_shengke['旺行']}）：红球 {b_ke_wo_red} ｜蓝球 {b_ke_wo_blue}",
            f"- 出生我克行（{b_shengke['旺行']}→{b_shengke['我克行']}）：{b_wo_ke_red}",
        ]
        formatted_shengke += "\n" + "\n".join(birth_lines)

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

    dimension_count = (11 if hour_zhi else 10) if (birthday and b_shengke) else (10 if hour_zhi else 9)
    _dim_names = "旺行+生我行+日月+月相+六柱干支+纳音五行+飞星方位+🌌值宿+🌌七曜"
    if hour_zhi:
        _dim_names += "+时辰"
    if birthday and b_shengke:
        _dim_names += "+🎂出生"
    summary_lines = [
        f"【综合号码热度汇总（娱乐）】",
        f"以下号码在{dimension_count}个维度（{_dim_names}）重合出现，⭐越多重合度越高：",
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
        f"💡 旺行判定模式：{mode_desc}" + (f"（{auto_reason}）" if auto_reason else ""),
    ])
    if birthday and b_day_gan:
        summary_lines.extend([
            f"🎂 出生维度模式：{b_day_gan}{b_day_zhi}日·{b_day_wuxing}行（仅供参考）",
        ])
    formatted_summary = "\n".join(summary_lines)

    result = {
        "formatted_shengke": formatted_shengke,
        "formatted_sun_moon": formatted_sun_moon,
        "formatted_moon_phase": formatted_moon_phase,
        "formatted_summary": formatted_summary,
        "formatted_liuzhu": formatted_liuzhu,
        "formatted_feixing": formatted_feixing,
        "formatted_xingxiu": formatted_xingxiu,
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


# ===== 双色球历史开奖数据+统计分析 =====

# 加载历史数据
import os as _os
_ssq_history_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ssq_history.json")
_SSQ_HISTORY = []
if _os.path.exists(_ssq_history_path):
    with open(_ssq_history_path, "r", encoding="utf-8") as _f:
        _SSQ_HISTORY = json.load(_f)


@app.get("/ssq/history", tags=["双色球历史数据"])
async def ssq_history(limit: int = 30):
    """
    双色球历史开奖数据查询

    - **limit**: 返回最近N期数据，默认30，最大200
    """
    if not _SSQ_HISTORY:
        raise HTTPException(status_code=503, detail="历史数据未加载")
    limit = min(limit, 200)
    return {
        "total": str(len(_SSQ_HISTORY)),
        "data": json.dumps(_SSQ_HISTORY[:limit], ensure_ascii=False)
    }


@app.get("/ssq/analysis", tags=["双色球历史数据"])
async def ssq_analysis(periods: int = 50):
    """
    双色球基础统计分析

    - **periods**: 分析最近N期数据，默认50，最大200

    返回：号码频率、遗漏值、冷热号、和值分布、奇偶比、大小比、区间分布
    """
    if not _SSQ_HISTORY:
        raise HTTPException(status_code=503, detail="历史数据未加载")
    periods = min(periods, len(_SSQ_HISTORY))
    data = _SSQ_HISTORY[:periods]

    # ===== 1. 号码频率统计 =====
    red_freq = {i: 0 for i in range(1, 34)}
    blue_freq = {i: 0 for i in range(1, 17)}

    for rec in data:
        for n in rec["red"]:
            red_freq[n] += 1
        blue_freq[rec["blue"]] += 1

    # 红球频率排名
    red_freq_sorted = sorted(red_freq.items(), key=lambda x: (-x[1], x[0]))
    # 蓝球频率排名
    blue_freq_sorted = sorted(blue_freq.items(), key=lambda x: (-x[1], x[0]))

    # ===== 2. 遗漏值（当前连续未出期数） =====
    red_miss = {i: 0 for i in range(1, 34)}
    blue_miss = {i: 0 for i in range(1, 17)}

    for num in range(1, 34):
        for rec in data:
            if num in rec["red"]:
                break
            red_miss[num] += 1

    for num in range(1, 17):
        for rec in data:
            if num == rec["blue"]:
                break
            blue_miss[num] += 1

    # 遗漏排名
    red_miss_sorted = sorted(red_miss.items(), key=lambda x: (-x[1], x[0]))
    blue_miss_sorted = sorted(blue_miss.items(), key=lambda x: (-x[1], x[0]))

    # ===== 3. 冷热号（近N期） =====
    avg_red = periods * 6 / 33
    avg_blue = periods / 16

    def _red_temp(freq):
        if freq >= avg_red * 1.5:
            return "🔥热"
        elif freq <= avg_red * 0.5:
            return "❄️冷"
        else:
            return "📐温"

    def _blue_temp(freq):
        if freq >= avg_blue * 1.5:
            return "🔥热"
        elif freq <= avg_blue * 0.5:
            return "❄️冷"
        else:
            return "📐温"

    red_temp = {n: _red_freq for n, _red_freq in red_freq.items()}
    blue_temp = {n: _blue_freq for n, _blue_freq in blue_freq.items()}

    # ===== 4. 和值统计 =====
    sum_values = [sum(rec["red"]) for rec in data]
    avg_sum = round(sum(sum_values) / len(sum_values), 1)
    min_sum = min(sum_values)
    max_sum = max(sum_values)

    # 和值区间分布
    sum_ranges = {"21-60": 0, "61-100": 0, "101-140": 0, "141-183": 0}
    for s in sum_values:
        if s <= 60:
            sum_ranges["21-60"] += 1
        elif s <= 100:
            sum_ranges["61-100"] += 1
        elif s <= 140:
            sum_ranges["101-140"] += 1
        else:
            sum_ranges["141-183"] += 1

    # ===== 5. 奇偶比 =====
    odd_even_counts = {}
    for rec in data:
        odd = sum(1 for n in rec["red"] if n % 2 == 1)
        even = 6 - odd
        ratio = f"{odd}:{even}"
        odd_even_counts[ratio] = odd_even_counts.get(ratio, 0) + 1
    odd_even_sorted = sorted(odd_even_counts.items(), key=lambda x: (-x[1], x[0]))

    # ===== 6. 大小比（1-16小，17-33大） =====
    big_small_counts = {}
    for rec in data:
        big = sum(1 for n in rec["red"] if n >= 17)
        small = 6 - big
        ratio = f"{big}:{small}"
        big_small_counts[ratio] = big_small_counts.get(ratio, 0) + 1
    big_small_sorted = sorted(big_small_counts.items(), key=lambda x: (-x[1], x[0]))

    # ===== 7. 区间分布（1-11/12-22/23-33） =====
    zone_counts = {"一区(01-11)": 0, "二区(12-22)": 0, "三区(23-33)": 0}
    for rec in data:
        for n in rec["red"]:
            if n <= 11:
                zone_counts["一区(01-11)"] += 1
            elif n <= 22:
                zone_counts["二区(12-22)"] += 1
            else:
                zone_counts["三区(23-33)"] += 1

    # ===== 8. 连号统计 =====
    cons_count = 0
    for rec in data:
        r = sorted(rec["red"])
        for i in range(len(r) - 1):
            if r[i + 1] - r[i] == 1:
                cons_count += 1
                break
    cons_ratio = round(cons_count / len(data) * 100, 1)

    # ===== 9. 重号统计（与上一期重复） =====
    repeat_count = 0
    repeat_detail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for i in range(len(data) - 1):
        cur = set(data[i]["red"])
        prev = set(data[i + 1]["red"])
        overlap = len(cur & prev)
        repeat_count += overlap
        if overlap in repeat_detail:
            repeat_detail[overlap] += 1
        elif overlap > 4:
            repeat_detail[4] += 1
    avg_repeat = round(repeat_count / max(len(data) - 1, 1), 2)

    # ===== 格式化输出 =====
    # 红球频率TOP10
    red_freq_top10 = "  ".join([f"{n:02d}({c}次)" for n, c in red_freq_sorted[:10]])
    # 红球频率BOTTOM10
    red_freq_bot10 = "  ".join([f"{n:02d}({c}次)" for n, c in red_freq_sorted[-10:]])
    # 蓝球频率TOP5
    blue_freq_top5 = "  ".join([f"{n:02d}({c}次)" for n, c in blue_freq_sorted[:5]])
    # 红球遗漏TOP10
    red_miss_top10 = "  ".join([f"{n:02d}({c}期)" for n, c in red_miss_sorted[:10]])
    # 蓝球遗漏TOP5
    blue_miss_top5 = "  ".join([f"{n:02d}({c}期)" for n, c in blue_miss_sorted[:5]])

    # 冷热号列表
    hot_red = sorted([n for n, f in red_freq.items() if f >= avg_red * 1.5])
    cold_red = sorted([n for n, f in red_freq.items() if f <= avg_red * 0.5])
    hot_blue = sorted([n for n, f in blue_freq.items() if f >= avg_blue * 1.5])
    cold_blue = sorted([n for n, f in blue_freq.items() if f <= avg_blue * 0.5])

    formatted_analysis = (
        f"【双色球基础统计分析（近{periods}期）】\n\n"
        f"📊 红球频率TOP10：{red_freq_top10}\n"
        f"📊 红球频率BOTTOM10：{red_freq_bot10}\n"
        f"📊 蓝球频率TOP5：{blue_freq_top5}\n\n"
        f"⏳ 红球遗漏TOP10：{red_miss_top10}\n"
        f"⏳ 蓝球遗漏TOP5：{blue_miss_top5}\n\n"
        f"🔥 红球热号：{', '.join(f'{n:02d}' for n in hot_red) if hot_red else '无'}\n"
        f"❄️ 红球冷号：{', '.join(f'{n:02d}' for n in cold_red) if cold_red else '无'}\n"
        f"🔥 蓝球热号：{', '.join(f'{n:02d}' for n in hot_blue) if hot_blue else '无'}\n"
        f"❄️ 蓝球冷号：{', '.join(f'{n:02d}' for n in cold_blue) if cold_blue else '无'}\n\n"
        f"📈 和值范围：{min_sum}~{max_sum}，平均{avg_sum}\n"
        f"📈 和值分布：21-60({sum_ranges['21-60']}期) 61-100({sum_ranges['61-100']}期) "
        f"101-140({sum_ranges['101-140']}期) 141-183({sum_ranges['141-183']}期)\n\n"
        f"⚖️ 奇偶比分布：{'  '.join([f'{r}({c}期)' for r, c in odd_even_sorted[:5]])}\n"
        f"⚖️ 大小比分布：{'  '.join([f'{r}({c}期)' for r, c in big_small_sorted[:5]])}\n\n"
        f"🗺️ 区间分布：{'  '.join([f'{k}({v}个)' for k, v in zone_counts.items()])}\n\n"
        f"🔗 连号出现率：{cons_ratio}%\n"
        f"🔄 重号平均：{avg_repeat}个/期，分布：{'  '.join([f'{k}个({v}期)' for k, v in sorted(repeat_detail.items()) if v > 0])}"
    )

    # ===== v4.0 新增：马尔可夫预测 + 衰减记忆 + 关联规则 =====
    # 马尔可夫链预测（基于近periods期转移矩阵）
    markov_trans_red = _markov_transition(data, "red")
    markov_trans_blue = _markov_transition(data, "blue")
    last_red = data[0]["red"]
    last_blue = [data[0]["blue"]]
    markov_pred_red = _markov_predict(markov_trans_red, last_red, "red")
    markov_pred_blue = _markov_predict(markov_trans_blue, last_blue, "blue")

    markov_red_top5 = sorted(markov_pred_red.items(), key=lambda x: -x[1])[:5]
    markov_blue_top3 = sorted(markov_pred_blue.items(), key=lambda x: -x[1])[:3]

    # 衰减记忆频率
    decay_freq_red = _decay_weighted_stats(data, "red", 0.95)
    decay_freq_blue = _decay_weighted_stats(data, "blue", 0.95)
    decay_red_top5 = sorted(decay_freq_red.items(), key=lambda x: -x[1])[:5]
    decay_blue_top3 = sorted(decay_freq_blue.items(), key=lambda x: -x[1])[:3]

    # 关联规则
    assoc_rules_red = _association_rules(data, "red", min_support=0.03, min_confidence=0.12)
    assoc_rules_blue = _association_rules(data, "blue", min_support=0.02, min_confidence=0.08)

    formatted_analysis += (
        f"\n\n【v4.0 统计引擎升级】\n\n"
        f"🔮 马尔可夫预测红球TOP5：{'  '.join([f'{n:02d}({s:.1%})' for n,s in markov_red_top5])}\n"
        f"🔮 马尔可夫预测蓝球TOP3：{'  '.join([f'{n:02d}({s:.1%})' for n,s in markov_blue_top3])}\n\n"
        f"📉 衰减频率红球TOP5：{'  '.join([f'{n:02d}({s:.2f})' for n,s in decay_red_top5])}\n"
        f"📉 衰减频率蓝球TOP3：{'  '.join([f'{n:02d}({s:.2f})' for n,s in decay_blue_top3])}\n\n"
        f"🔗 关联规则(红球)：{len(assoc_rules_red)}条有效规则，TOP3：{'  '.join([f'{a}→{b}({c:.0%})' for a,b,c,_,_ in assoc_rules_red[:3]])}\n"
        f"🔗 关联规则(蓝球)：{len(assoc_rules_blue)}条有效规则，TOP3：{'  '.join([f'红{a}→蓝{b}({c:.0%})' for a,b,c,_,_ in assoc_rules_blue[:3]])}"
    )

    # v4.1 冷热周期识别
    hot_cold_red_a = _hot_cold_cycle(data, "red", window=10)
    hot_cold_blue_a = _hot_cold_cycle(data, "blue", window=10)

    # 冷热周期摘要
    cold_to_warm_r = [n for n in range(1,34) if hot_cold_red_a[n]["turn"] == "冷→温↑"]
    warm_to_hot_r = [n for n in range(1,34) if hot_cold_red_a[n]["turn"] == "温→热↑"]
    hot_stable_r = [n for n in range(1,34) if hot_cold_red_a[n]["turn"] == "热→"]
    hot_cooling_r = [n for n in range(1,34) if hot_cold_red_a[n]["turn"] == "热→温↓"]
    cold_stable_r = [n for n in range(1,34) if hot_cold_red_a[n]["turn"] == "冷→"]
    warm_cooling_r = [n for n in range(1,34) if hot_cold_red_a[n]["turn"] == "温→冷↓"]

    cold_to_warm_b = [n for n in range(1,17) if hot_cold_blue_a[n]["turn"] == "冷→温↑"]
    warm_to_hot_b = [n for n in range(1,17) if hot_cold_blue_a[n]["turn"] == "温→热↑"]
    hot_stable_b = [n for n in range(1,17) if hot_cold_blue_a[n]["turn"] == "热→"]
    hot_cooling_b = [n for n in range(1,17) if hot_cold_blue_a[n]["turn"] == "热→温↓"]

    formatted_analysis += (
        f"\n\n【v4.1 冷热周期识别】\n\n"
        f"🌡️ 红球冷转热（追冷回补⭐）：{'  '.join([f'{n:02d}' for n in sorted(cold_to_warm_r)])}\n"
        f"🌡️ 红球温转热（升温中）：{'  '.join([f'{n:02d}' for n in sorted(warm_to_hot_r)])}\n"
        f"🌡️ 红球稳定热号：{'  '.join([f'{n:02d}' for n in sorted(hot_stable_r)])}\n"
        f"🌡️ 红球热转冷（警惕⚠️）：{'  '.join([f'{n:02d}' for n in sorted(hot_cooling_r)])}\n"
        f"🌡️ 红球温转冷：{'  '.join([f'{n:02d}' for n in sorted(warm_cooling_r)])}\n"
        f"🌡️ 红球稳定冷号：{'  '.join([f'{n:02d}' for n in sorted(cold_stable_r)])}\n\n"
        f"🔵 蓝球冷转热⭐：{'  '.join([f'{n:02d}' for n in sorted(cold_to_warm_b)])}\n"
        f"🔵 蓝球温转热：{'  '.join([f'{n:02d}' for n in sorted(warm_to_hot_b)])}\n"
        f"🔵 蓝球稳定热号：{'  '.join([f'{n:02d}' for n in sorted(hot_stable_b)])}\n"
        f"🔵 蓝球热转冷⚠️：{'  '.join([f'{n:02d}' for n in sorted(hot_cooling_b)])}"
    )

    return {
        "periods_analyzed": periods,
        "date_range": f"{data[-1]['date']} ~ {data[0]['date']}" if data else "",
        "formatted_analysis": formatted_analysis,
        # 结构化数据（供程序调用）
        "red_freq": {str(k): v for k, v in sorted(red_freq.items())},
        "blue_freq": {str(k): v for k, v in sorted(blue_freq.items())},
        "red_miss": {str(k): v for k, v in sorted(red_miss.items())},
        "blue_miss": {str(k): v for k, v in sorted(blue_miss.items())},
        "avg_sum": avg_sum,
        "sum_range": [min_sum, max_sum],
        "consecutive_ratio": cons_ratio,
        "avg_repeat": avg_repeat,
    }


# ===== v4.0 统计引擎升级：马尔可夫链 + 衰减记忆 + 关联规则 =====

def _markov_transition(data, ball_type="red"):
    """
    P1: 一阶马尔可夫链转移矩阵

    统计历史开奖中「当前期出i → 下期出j」的转移概率。
    返回: {i: {j: probability}} 归一化转移概率矩阵

    ball_type: "red" 红球(1-33) / "blue" 蓝球(1-16)
    data: 历史开奖列表（最新期在前）
    """
    if ball_type == "red":
        nums = range(1, 34)
    else:
        nums = range(1, 17)

    # 统计转移计数：data是最新期在前，data[i+1]是前一期的下一期（更早）
    # 所以 data[i] → data[i+1] 表示"从新一期到更早一期"，但我们关心的是"从上一期预测下一期"
    # 正确方向：data[i+1](更早) → data[i](更新)，即 prev→current
    trans_count = {i: {j: 0 for j in nums} for i in nums}
    trans_total = {i: 0 for i in nums}

    for idx in range(len(data) - 1):
        prev_rec = data[idx + 1]  # 更早的一期（上一期）
        curr_rec = data[idx]      # 更新的一期（下一期）

        if ball_type == "red":
            prev_balls = prev_rec["red"]
            curr_balls = curr_rec["red"]
        else:
            prev_balls = [prev_rec["blue"]]
            curr_balls = [curr_rec["blue"]]

        for p in prev_balls:
            trans_total[p] += 1
            for c in curr_balls:
                trans_count[p][c] += 1

    # 归一化为概率
    trans_prob = {}
    for i in nums:
        total = trans_total[i]
        if total > 0:
            trans_prob[i] = {j: trans_count[i][j] / total for j in nums}
        else:
            trans_prob[i] = {j: 1.0 / len(list(nums)) for j in nums}

    return trans_prob


def _markov_predict(trans_prob, last_numbers, ball_type="red"):
    """
    基于马尔可夫转移矩阵预测下期各号码出现概率。

    last_numbers: 上一期开出的号码列表
    返回: {num: probability} 各号码的预测概率
    """
    if ball_type == "red":
        nums = range(1, 34)
    else:
        nums = range(1, 17)

    # 对于每个候选号码，累加从上一期各号码转移过来的概率
    scores = {n: 0.0 for n in nums}
    for prev in last_numbers:
        if prev in trans_prob:
            for n in nums:
                scores[n] += trans_prob[prev].get(n, 0)

    # 归一化
    total = sum(scores.values())
    if total > 0:
        scores = {n: s / total for n, s in scores.items()}

    return scores


def _decay_weighted_stats(data, ball_type="red", decay=0.95):
    """
    P2: 衰减记忆加权统计

    近期数据权重更高（指数衰减），远期数据权重递减。
    替代当前等权统计（频率60%+遗漏40%）。

    decay: 衰减系数，0.95表示每期权重乘0.95
    data: 历史开奖列表（最新期在前，index0=最新）
    返回: {num: weighted_score} 归一化到0-1的衰减加权分
    """
    if ball_type == "red":
        nums = range(1, 34)
    else:
        nums = range(1, 17)

    weighted_freq = {n: 0.0 for n in nums}
    # data[0]最新 → 权重1.0（decay^0），data[1] → decay^1, ...
    for idx, rec in enumerate(data):
        w = decay ** idx  # 指数衰减
        if ball_type == "red":
            for n in rec["red"]:
                weighted_freq[n] += w
        else:
            weighted_freq[rec["blue"]] += w

    # 归一化到0-1
    max_w = max(weighted_freq.values()) if weighted_freq.values() else 1
    if max_w == 0:
        max_w = 1
    scores = {n: weighted_freq[n] / max_w for n in nums}

    return scores


def _decay_miss_with_weight(data, ball_type="red", decay=0.95):
    """
    P2补充: 衰减遗漏值 - 越久没出且权重高 → 越应该出

    与传统遗漏不同，这里加入衰减因子：近期遗漏权重更高。
    返回: {num: weighted_miss_score} 归一化到0-1
    """
    if ball_type == "red":
        nums = range(1, 34)
    else:
        nums = range(1, 17)

    miss_score = {n: 0.0 for n in nums}

    for n in nums:
        # 找到该号码最近一次出现的位置
        for idx, rec in enumerate(data):
            if ball_type == "red" and n in rec["red"]:
                # 出现了，遗漏期数=idx，权重=decay^idx
                miss_score[n] = decay ** idx
                break
            elif ball_type == "blue" and rec["blue"] == n:
                miss_score[n] = decay ** idx
                break
        else:
            # 全部都没出现，给最低分
            miss_score[n] = 0.0

    # 反转：遗漏越久（miss_score越低）→ 补回概率越高
    # 但衰减遗漏的逻辑是：已经很久没出的号码，如果近期权重高说明"该出了"
    # 实际上miss_score=decay^idx，idx=遗漏期数，遗漏越久miss_score越低
    # 我们要的是"遗漏越久越可能出"，所以反转
    inverted = {n: 1.0 - miss_score[n] for n in nums}

    # 归一化
    max_inv = max(inverted.values()) if inverted.values() else 1
    if max_inv == 0:
        max_inv = 1
    scores = {n: inverted[n] / max_inv for n in nums}

    return scores


def _association_rules(data, ball_type="red", min_support=0.03, min_confidence=0.12):
    """
    P3: 关联规则挖掘（Apriori简化版）

    挖掘号码共现模式：如果出A则大概率出B。
    对于红球：挖掘"如果本期出A则下期出B"的转移关联规则
    对于蓝球：挖掘"如果本期红球出A则下期蓝球出B"的跨维度规则

    min_support: 最小支持度（规则在所有期中出现的最低比例）
    min_confidence: 最小置信度（P(B|A)的最低值）
    返回: [(antecedent, consequent, confidence, support)] 排序后的规则列表
    """
    rules = []

    if ball_type == "red":
        # 红球→红球转移关联：本期出A → 下期出B
        pair_count = {}  # {(a, b): count}
        a_count = {}     # {a: count} 前件出现次数

        for idx in range(len(data) - 1):
            curr_red = data[idx]["red"]  # 当前期
            prev_red = data[idx + 1]["red"]  # 上一期（更早）

            for a in prev_red:
                a_count[a] = a_count.get(a, 0) + 1
                for b in curr_red:
                    key = (a, b)
                    pair_count[key] = pair_count.get(key, 0) + 1

        total_periods = len(data) - 1

        for (a, b), cnt in pair_count.items():
            support = cnt / total_periods
            confidence = cnt / a_count[a] if a_count[a] > 0 else 0
            # 过滤：支持度>min_support 且 置信度>min_confidence 且 高于随机概率
            random_prob = 6 / 33  # 红球随机概率≈18.18%
            if support >= min_support and confidence >= min_confidence and confidence > random_prob:
                lift = confidence / random_prob  # 提升度
                rules.append((a, b, round(confidence, 4), round(support, 4), round(lift, 4)))

    elif ball_type == "blue":
        # 红球→蓝球转移关联：本期红球出A → 下期蓝球出B
        pair_count = {}
        a_count = {}

        for idx in range(len(data) - 1):
            curr_blue = data[idx]["blue"]
            prev_red = data[idx + 1]["red"]

            for a in prev_red:
                a_count[a] = a_count.get(a, 0) + 1
                key = (a, curr_blue)
                pair_count[key] = pair_count.get(key, 0) + 1

        total_periods = len(data) - 1

        for (a, b), cnt in pair_count.items():
            support = cnt / total_periods
            confidence = cnt / a_count[a] if a_count[a] > 0 else 0
            random_prob = 1 / 16  # 蓝球随机概率≈6.25%
            if support >= min_support and confidence >= min_confidence and confidence > random_prob:
                lift = confidence / random_prob
                rules.append((a, b, round(confidence, 4), round(support, 4), round(lift, 4)))

    # 按置信度降序排列
    rules.sort(key=lambda x: (-x[2], -x[3]))
    return rules


def _association_predict(rules, last_numbers, ball_type="red"):
    """
    基于关联规则预测下期各号码得分。

    last_numbers: 上一期开出的号码列表（红球或蓝球）
    返回: {num: score} 归一化到0-1的关联规则预测分
    """
    if ball_type == "red":
        nums = range(1, 34)
    else:
        nums = range(1, 17)

    scores = {n: 0.0 for n in nums}

    for rule in rules:
        antecedent, consequent, confidence, support, lift = rule
        if antecedent in last_numbers:
            scores[consequent] += confidence * lift  # 置信度×提升度作为权重

    # 归一化到0-1
    max_s = max(scores.values()) if scores.values() else 1
    if max_s == 0:
        max_s = 1
    scores = {n: s / max_s for n, s in scores.items()}

    return scores


def _hot_cold_cycle(data, ball_type="red", window=10):
    """
    v4.1 P4: 冷热周期识别引擎

    用滑动窗口统计每个号码的冷热状态和转换拐点。
    - window: 滑动窗口期数，默认10期
    返回: {
        num: {
            "status": "hot"/"warm"/"cold",       # 当前冷热状态
            "trend": "rising"/"falling"/"stable",  # 趋势方向
            "freq_recent": float,                  # 近window期出现频率
            "freq_prev": float,                    # 前window期出现频率（对比用）
            "turn": str,                           # 拐点描述（如"冷→温↑"）
            "score": float,                        # 0-1冷热周期得分（冷转热得分最高）
        }
    }
    """
    if ball_type == "red":
        nums = range(1, 34)
    else:
        nums = range(1, 17)

    total = len(data)
    if total < window * 2:
        # 数据不足，返回默认
        return {n: {"status": "warm", "trend": "stable", "freq_recent": 0,
                     "freq_prev": 0, "turn": "数据不足", "score": 0.5} for n in nums}

    # 近window期 vs 前window期对比
    recent_data = data[:window]
    prev_data = data[window:window*2]

    result = {}
    for n in nums:
        # 近期频率
        freq_recent = sum(1 for rec in recent_data
                          if n in (rec["red"] if ball_type == "red" else [rec["blue"]])) / window
        # 前期频率
        freq_prev = sum(1 for rec in prev_data
                        if n in (rec["red"] if ball_type == "red" else [rec["blue"]])) / window

        # 冷热状态判定（基于近期频率）
        if ball_type == "red":
            # 红球期望频率 = 6/33 ≈ 0.182
            if freq_recent >= 0.25:
                status = "hot"
            elif freq_recent >= 0.10:
                status = "warm"
            else:
                status = "cold"
        else:
            # 蓝球期望频率 = 1/16 ≈ 0.0625
            if freq_recent >= 0.12:
                status = "hot"
            elif freq_recent >= 0.04:
                status = "warm"
            else:
                status = "cold"

        # 趋势判定（近期vs前期频率变化）
        delta = freq_recent - freq_prev
        if delta > 0.08:
            trend = "rising"
        elif delta < -0.08:
            trend = "falling"
        else:
            trend = "stable"

        # 拐点描述
        status_cn = {"hot": "热", "warm": "温", "cold": "冷"}
        trend_cn = {"rising": "↑", "falling": "↓", "stable": "→"}
        turn = f"{status_cn[status]}{trend_cn[trend]}"

        # 特殊拐点标记
        if status == "cold" and trend == "rising":
            turn = "冷→温↑"  # 冷转热拐点！最有价值
        elif status == "hot" and trend == "falling":
            turn = "热→温↓"  # 热转冷拐点，需警惕
        elif status == "warm" and trend == "rising":
            turn = "温→热↑"  # 正在升温
        elif status == "warm" and trend == "falling":
            turn = "温→冷↓"  # 正在降温

        # 冷热周期得分（0-1）
        # 核心逻辑：冷转热得分最高（追冷回补），热且稳定次之（追热），热转冷最低（避开）
        if status == "cold" and trend == "rising":
            score = 0.95  # 冷转热拐点，最高分
        elif status == "warm" and trend == "rising":
            score = 0.80  # 温转热
        elif status == "hot" and trend == "stable":
            score = 0.70  # 稳定热号
        elif status == "hot" and trend == "rising":
            score = 0.65  # 持续升温
        elif status == "warm" and trend == "stable":
            score = 0.50  # 中性温号
        elif status == "cold" and trend == "stable":
            score = 0.40  # 稳定冷号
        elif status == "warm" and trend == "falling":
            score = 0.30  # 温转冷
        elif status == "cold" and trend == "falling":
            score = 0.15  # 极冷下降
        elif status == "hot" and trend == "falling":
            score = 0.25  # 热转冷
        else:
            score = 0.50

        result[n] = {
            "status": status,
            "trend": trend,
            "freq_recent": round(freq_recent, 3),
            "freq_prev": round(freq_prev, 3),
            "turn": turn,
            "score": score,
        }

    return result


def _adaptive_engine_weights(data, window=30):
    """
    v4.1 P8: 自适应引擎权重

    根据最近window期各统计引擎的预测准确度，动态调整权重。
    - window: 评估窗口期数，默认30期
    返回: {"decay_freq": float, "decay_miss": float, "markov": float, "association": float}
    """
    total = len(data)
    if total < window + 10:
        # 数据不足，使用默认权重
        return {"decay_freq": 0.30, "decay_miss": 0.20, "markov": 0.30, "association": 0.20}

    # 对最近window期，每期用前10期数据预测，计算各引擎命中率
    engine_hits = {"decay_freq": 0, "decay_miss": 0, "markov": 0, "association": 0}
    engine_total = 0

    eval_data = data[:window]
    train_base = 10  # 每次预测用的训练数据期数

    for i in range(len(eval_data) - 1):
        actual_red = set(eval_data[i]["red"])
        actual_blue = {eval_data[i]["blue"]}

        # 训练数据：从当前期往后取train_base期（数据是倒序的）
        train_start = i + 1
        train_end = min(i + 1 + train_base, total)
        if train_end - train_start < 5:
            continue

        train = data[train_start:train_end]
        last_red = data[train_start]["red"]  # 上一期红球
        last_blue = [data[train_start]["blue"]]

        # 引擎1: 衰减频率
        df_red = _decay_weighted_stats(train, "red", decay=0.95)
        df_top6 = set(sorted(df_red, key=df_red.get, reverse=True)[:6])

        # 引擎1b: 衰减遗漏
        dm_red = _decay_miss_with_weight(train, "red", decay=0.95)
        dm_top6 = set(sorted(dm_red, key=dm_red.get, reverse=True)[:6])

        # 引擎2: 马尔可夫
        mt_red = _markov_transition(train, "red")
        mp_red = _markov_predict(mt_red, last_red, "red")
        mk_top6 = set(sorted(mp_red, key=mp_red.get, reverse=True)[:6])

        # 引擎3: 关联规则
        ar_red = _association_rules(train, "red", min_support=0.03, min_confidence=0.12)
        ap_red = _association_predict(ar_red, last_red, "red")
        as_top6 = set(sorted(ap_red, key=ap_red.get, reverse=True)[:6])

        # 命中判定：TOP6与实际交集≥2个算命中
        if len(df_top6 & actual_red) >= 2:
            engine_hits["decay_freq"] += 1
        if len(dm_top6 & actual_red) >= 2:
            engine_hits["decay_miss"] += 1
        if len(mk_top6 & actual_red) >= 2:
            engine_hits["markov"] += 1
        if len(as_top6 & actual_red) >= 2:
            engine_hits["association"] += 1

        engine_total += 1

    if engine_total == 0:
        return {"decay_freq": 0.30, "decay_miss": 0.20, "markov": 0.30, "association": 0.20}

    # 计算各引擎命中率
    hit_rates = {k: v / engine_total for k, v in engine_hits.items()}

    # 基于命中率分配权重（命中率越高权重越大，保底10%）
    total_hit = sum(hit_rates.values())
    if total_hit == 0:
        return {"decay_freq": 0.30, "decay_miss": 0.20, "markov": 0.30, "association": 0.20}

    weights = {}
    for engine, rate in hit_rates.items():
        weights[engine] = max(0.10, rate / total_hit)  # 保底10%

    # 归一化到总和=1
    w_sum = sum(weights.values())
    weights = {k: round(v / w_sum, 2) for k, v in weights.items()}

    return weights


@ app.get("/ssq/backtest", tags=["双色球历史数据"])
async def ssq_backtest(periods: int = 200, mode: str = "day_gan", birthday: str = ""):
    """
    双色球玄学维度回测验证

    对历史每期数据，用/ganzhi相同逻辑计算各维度号码，与实际开奖号码对比。
    - **periods**: 回测最近N期，默认200，最大2144
    - **mode**: 旺行判定逻辑，可选：
      - day_gan（默认）：日柱天干五行
      - day_zhi：日柱地支五行
      - majority：六柱综合众数
      - all：三种模式并行回测，对比结果并推荐最优

    返回每个维度在红球/蓝球中的命中率，与随机概率对比。
    当mode="all"时，额外返回三种模式的对比结果和推荐。
    """
    if not _SSQ_HISTORY:
        raise HTTPException(status_code=503, detail="历史数据未加载")
    periods = min(periods, len(_SSQ_HISTORY))
    data = _SSQ_HISTORY[:periods]

    if mode not in ("day_gan", "day_zhi", "majority", "all"):
        raise HTTPException(
            status_code=400,
            detail="mode参数错误，可选：day_gan / day_zhi / majority / all"
        )

    # 如果mode="all"，并行跑三种模式
    if mode == "all":
        results_all = {}
        for m in ["day_gan", "day_zhi", "majority"]:
            result = await _run_backtest(data, periods, m, birthday)
            results_all[m] = result
        # 对比三种模式，生成推荐
        recommend = _compare_backtest_modes(results_all, periods)
        return {
            "periods_tested": periods,
            "mode": "all",
            "results_all": results_all,
            "formatted_backtest_all": recommend["formatted"],
            "recommend_mode": recommend["recommend_mode"],
            "recommend_reason": recommend["reason"],
        }

    # 单模式回测
    result = await _run_backtest(data, periods, mode, birthday)
    return result


async def _run_backtest(data, periods, mode, birthday=""):
    """
    内部函数：对指定数据和mode执行回测
    """
    dim_stats = {
        "旺行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "生我行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "我生行·泄": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "克我行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "我克行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "纳音五行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "六柱干支": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "飞星方位": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🌌值宿": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🌌七曜": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🎂出生旺行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🎂出生生我行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🎂出生克我行": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🎂出生我生行·泄": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
        "🎂出生日柱": {"red_hit": 0, "red_total": 0, "blue_hit": 0, "blue_total": 0, "red_pool": 0, "blue_pool": 0},
    }

    _TIANGAN_LIST = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    _DIZHI_LIST = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    month_dz_map_bt = {1: '丑', 2: '寅', 3: '卯', 4: '辰', 5: '巳', 6: '午',
                      7: '未', 8: '申', 9: '酉', 10: '戌', 11: '亥', 12: '子'}
    tg_start_map_bt = {'甲': '丙', '己': '丙', '乙': '戊', '庚': '戊', '丙': '庚', '辛': '庚',
                       '丁': '壬', '壬': '壬', '戊': '甲', '癸': '甲'}
    month_dz_order_bt = ['寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥', '子', '丑']

    from datetime import date as date_cls
    from lunarcalendar import Converter, Solar
    from collections import Counter

    base_date = date_cls(2000, 1, 7)

    for rec in data:
        date_str = rec["date"]
        if not date_str:
            continue
        # 兼容日期格式：清理中文星期如"2020-01-21(二)"→"2020-01-21"
        clean_date = date_str.split('(')[0] if '(' in date_str else date_str
        parts = clean_date.split('-')
        try:
            solar_date = date_cls(int(parts[0]), int(parts[1]), int(parts[2]))
        except:
            continue

        # 干支计算
        y_offset = solar_date.year - 1984
        year_gan = _TIANGAN_LIST[y_offset % 10]
        year_zhi = _DIZHI_LIST[y_offset % 12]
        diff = (solar_date - base_date).days
        day_gan = _TIANGAN_LIST[diff % 10]
        day_zhi = _DIZHI_LIST[diff % 12]
        month_zhi = month_dz_map_bt[solar_date.month]
        start_tg = tg_start_map_bt[year_gan]
        start_idx = _TIANGAN_LIST.index(start_tg)
        month_dz_idx = month_dz_order_bt.index(month_zhi)
        month_gan = _TIANGAN_LIST[(start_idx + month_dz_idx) % 10]

        # 旺行判定
        six_wx = [
            _TIANGAN_MAP[year_gan]["wuxing"], _DIZHI_RED_MAP[year_zhi]["wuxing"],
            _TIANGAN_MAP[month_gan]["wuxing"], _DIZHI_RED_MAP[month_zhi]["wuxing"],
            _TIANGAN_MAP[day_gan]["wuxing"], _DIZHI_RED_MAP[day_zhi]["wuxing"],
        ]
        wx_counter = Counter(six_wx)

        if mode == "day_zhi":
            day_wuxing = _DIZHI_RED_MAP[day_zhi]["wuxing"]
        elif mode == "majority":
            day_wuxing = wx_counter.most_common(1)[0][0]
        else:
            day_wuxing = _TIANGAN_MAP[day_gan]["wuxing"]

        shengke = _get_shengke_info(day_wuxing)

        # 阴历
        try:
            solar = Solar(solar_date.year, solar_date.month, solar_date.day)
            lunar = Converter.Solar2Lunar(solar)
            lunar_day = lunar.day
        except:
            continue

        # 实际开奖号码
        actual_red = set(rec["red"])
        actual_blue = rec["blue"]

        # 逐维度统计
        def _count_dim(dim_name, red_set, blue_set=None):
            dim_stats[dim_name]["red_hit"] += len(actual_red & red_set)
            dim_stats[dim_name]["red_total"] += 6
            dim_stats[dim_name]["red_pool"] = len(red_set)
            if blue_set is not None:
                dim_stats[dim_name]["blue_hit"] += 1 if actual_blue in blue_set else 0
                dim_stats[dim_name]["blue_total"] += 1
                dim_stats[dim_name]["blue_pool"] = len(blue_set)

        # 旺行
        _count_dim("旺行",
                    set(_WUXING_MAP[shengke["旺行"]]["red_balls"]),
                    set(_WUXING_MAP[shengke["旺行"]]["blue_balls"]))
        # 生我行
        _count_dim("生我行",
                    set(_WUXING_MAP[shengke["生我行"]]["red_balls"]),
                    set(_WUXING_MAP[shengke["生我行"]]["blue_balls"]))
        # 我生行·泄
        _count_dim("我生行·泄",
                    set(_WUXING_MAP[shengke["我生行(泄)"]]["red_balls"]))
        # 克我行
        _count_dim("克我行",
                    set(_WUXING_MAP[shengke["克我行"]]["red_balls"]),
                    set(_WUXING_MAP[shengke["克我行"]]["blue_balls"]))
        # 我克行
        _count_dim("我克行",
                    set(_WUXING_MAP[shengke["我克行"]]["red_balls"]))

        # 纳音五行
        day_ganzhi = day_gan + day_zhi
        day_nayin = _NAYIN_MAP.get(day_ganzhi, "")
        nayin_wuxing = _NAYIN_WUXING.get(day_nayin, "")
        if nayin_wuxing:
            _count_dim("纳音五行",
                        set(_WUXING_MAP[nayin_wuxing]["red_balls"]),
                        set(_WUXING_MAP[nayin_wuxing]["blue_balls"]))

        # 六柱干支
        liuzhu_red = set()
        liuzhu_blue = set()
        for tg, dz in [(year_gan, year_zhi), (month_gan, month_zhi), (day_gan, day_zhi)]:
            liuzhu_red.update(_TIANGAN_MAP[tg]["red_balls"])
            liuzhu_red.update(_DIZHI_RED_MAP[dz]["red_balls"])
            liuzhu_blue.add(_DIZHI_BLUE_MAP[dz])
        _count_dim("六柱干支", liuzhu_red, liuzhu_blue)

        # 飞星方位
        bagua = _DIZHI_BAGUA_MAP.get(day_zhi, {})
        if bagua:
            _count_dim("飞星方位",
                        set(bagua["red_balls"]),
                        set(bagua["blue_balls"]))

        # v5.0 P16: 二十八宿值宿回测
        zhixiu = _get_zhixiu(solar_date)
        xiu_wx = zhixiu["wuxing"]
        _count_dim("🌌值宿",
                    set(_WUXING_MAP[xiu_wx]["red_balls"]),
                    set(_WUXING_MAP[xiu_wx]["blue_balls"]))

        # v5.0 P16: 七曜回测
        qiyao = _get_qiyao(solar_date)
        yao_wx = qiyao["wuxing"]
        _count_dim("🌌七曜",
                    set(_WUXING_MAP[yao_wx]["red_balls"]),
                    set(_WUXING_MAP[yao_wx]["blue_balls"]))

        # v3.5: 出生维度回测（birthday参数）
        if birthday:
            try:
                b_parts = birthday.split('-')
                b_date = date_cls(int(b_parts[0]), int(b_parts[1]), int(b_parts[2]))
                b_diff = (b_date - base_date).days
                b_day_gan = _TIANGAN_LIST[b_diff % 10]
                b_day_zhi = _DIZHI_LIST[b_diff % 12]
                b_day_wuxing = _TIANGAN_MAP[b_day_gan]["wuxing"]
                b_shengke = _get_shengke_info(b_day_wuxing)
                # 出生旺行
                _count_dim("🎂出生旺行",
                            set(_WUXING_MAP[b_shengke["旺行"]]["red_balls"]),
                            set(_WUXING_MAP[b_shengke["旺行"]]["blue_balls"]))
                # 出生生我行
                _count_dim("🎂出生生我行",
                            set(_WUXING_MAP[b_shengke["生我行"]]["red_balls"]),
                            set(_WUXING_MAP[b_shengke["生我行"]]["blue_balls"]))
                # 出生克我行
                _count_dim("🎂出生克我行",
                            set(_WUXING_MAP[b_shengke["克我行"]]["red_balls"]),
                            set(_WUXING_MAP[b_shengke["克我行"]]["blue_balls"]))
                # 出生我生行·泄
                _count_dim("🎂出生我生行·泄",
                            set(_WUXING_MAP[b_shengke["我生行(泄)"]]["red_balls"]))
                # 出生日柱干支
                b_liuzhu_red = set()
                b_liuzhu_red.update(_TIANGAN_MAP[b_day_gan]["red_balls"])
                b_liuzhu_red.update(_DIZHI_RED_MAP[b_day_zhi]["red_balls"])
                b_liuzhu_blue = {_DIZHI_BLUE_MAP[b_day_zhi]}
                _count_dim("🎂出生日柱", b_liuzhu_red, b_liuzhu_blue)
            except:
                pass  # birthday参数错误时静默忽略

    # 计算命中率
    results = []
    for dim_name, stats in dim_stats.items():
        if stats["red_total"] == 0:
            continue
        red_hit_rate = round(stats["red_hit"] / stats["red_total"] * 100, 2)
        red_expected = round(stats["red_pool"] / 33 * 100, 2) if stats["red_pool"] > 0 else 0
        red_lift = round(red_hit_rate - red_expected, 2)

        blue_hit_rate = round(stats["blue_hit"] / stats["blue_total"] * 100, 2) if stats["blue_total"] > 0 else 0
        blue_expected = round(stats["blue_pool"] / 16 * 100, 2) if stats["blue_pool"] > 0 else 0
        blue_lift = round(blue_hit_rate - blue_expected, 2)

        if red_lift > 2:
            verdict = "✅有效"
        elif red_lift < -2:
            verdict = "❌负面"
        else:
            verdict = "⚠️中性"

        results.append({
            "dimension": dim_name,
            "red_hit": stats["red_hit"],
            "red_total": stats["red_total"],
            "red_hit_rate": red_hit_rate,
            "red_expected_rate": red_expected,
            "red_lift": red_lift,
            "blue_hit": stats["blue_hit"],
            "blue_total": stats["blue_total"],
            "blue_hit_rate": blue_hit_rate,
            "blue_expected_rate": blue_expected,
            "blue_lift": blue_lift,
            "verdict": verdict,
        })

    # 格式化输出
    lines = [f"【双色球玄学维度回测验证（近{periods}期，模式={mode}）】", ""]
    lines.append(f"基准：红球随机命中率≈{round(6 / 33 * 100, 2)}%/球，蓝球随机命中率≈{round(1 / 16 * 100, 2)}%")
    lines.append(f"提升值=实际命中率-期望命中率，>0=优于随机，<0=劣于随机")
    lines.append("")
    lines.append(f"{'维度':<10} {'红球命中':>8} {'红球命中率':>8} {'期望率':>6} {'提升':>6} {'蓝球命中率':>8} {'蓝球提升':>6} {'判定'}")
    lines.append("-" * 80)

    for r in results:
        blue_info = f"{r['blue_hit_rate']}%" if r['blue_total'] > 0 else "N/A"
        blue_lift_info = f"{r['blue_lift']}%" if r['blue_total'] > 0 else "N/A"
        lines.append(
            f"{r['dimension']:<10} {r['red_hit']:>5}/{r['red_total']:>3} "
            f"{r['red_hit_rate']:>7}% {r['red_expected_rate']:>5}% {r['red_lift']:>+5}% "
            f"{blue_info:>8} {blue_lift_info:>6} {r['verdict']}"
        )

    lines.append("")
    lines.append("💡 提升值>2%=有效维度，<-2%=负面维度，其余≈随机")

    formatted_backtest = "\n".join(lines)

    return {
        "periods_tested": periods,
        "mode": mode,
        "formatted_backtest": formatted_backtest,
        "details": results,
    }


def _compare_backtest_modes(results_all, periods):
    """
    对比三种模式的回测结果，生成formatted输出和推荐
    """
    mode_scores = {}
    for mode, result in results_all.items():
        details = result.get("details", [])
        red_lift_sum = sum(d["red_lift"] for d in details)
        blue_lift_sum = sum(d["blue_lift"] for d in details if d["blue_total"] > 0)
        valid_count = sum(1 for d in details if d["verdict"] == "✅有效")
        negative_count = sum(1 for d in details if d["verdict"] == "❌负面")
        mode_scores[mode] = {
            "red_lift_sum": red_lift_sum,
            "blue_lift_sum": blue_lift_sum,
            "valid_count": valid_count,
            "negative_count": negative_count,
            "score": red_lift_sum + blue_lift_sum * 0.5,
        }

    sorted_modes = sorted(mode_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    best_mode = sorted_modes[0][0]
    best_score = sorted_modes[0][1]

    lines = [f"【多模式回测对比（近{periods}期）】", ""]
    lines.append(f"{'模式':<12} {'红球提升∑':>10} {'蓝球提升∑':>10} {'有效维度':>8} {'负面维度':>8} {'综合得分':>8}")
    lines.append("-" * 70)

    mode_names = {"day_gan": "日干模式", "day_zhi": "日支模式", "majority": "六柱众数"}
    for mode, scores in sorted_modes:
        lines.append(
            f"{mode_names.get(mode, mode):<12} "
            f"{scores['red_lift_sum']:>+8}% "
            f"{scores['blue_lift_sum']:>+8}% "
            f"{scores['valid_count']:>6}个 "
            f"{scores['negative_count']:>6}个 "
            f"{scores['score']:>+7.2f}"
        )

    lines.append("")
    lines.append(f"🏆 推荐模式：{mode_names.get(best_mode, best_mode)}")
    lines.append(f"   综合得分最高（{best_score['score']:+.2f}），红球提升∑{best_score['red_lift_sum']:+.2f}%，有效维度{best_score['valid_count']}个")

    lines.append("")
    lines.append("📊 各模式有效维度（提升值>2%）：")
    for mode, result in results_all.items():
        valid_dims = [d for d in result.get("details", []) if d["verdict"] == "✅有效"]
        dim_str = "、".join([d["dimension"] for d in valid_dims]) if valid_dims else "无"
        lines.append(f"  {mode_names.get(mode, mode)}：{dim_str}")

    lines.append("")
    lines.append("💡 使用建议：")
    lines.append(f"  1. 在/ssq/pick接口中使用 mode={best_mode} 参数")
    lines.append(f"  2. 融合选号将自动采用「{mode_names.get(best_mode, best_mode)}」的计算结果")

    formatted = "\n".join(lines)

    return {
        "formatted": formatted,
        "recommend_mode": best_mode,
        "reason": f"综合得分最高（{best_score['score']:+.2f}），红球提升∑{best_score['red_lift_sum']:+.2f}%",
    }

@app.get("/ssq/adversarial", tags=["双色球历史数据"])
async def ssq_adversarial(mode: str = "day_zhi", birthday: str = ""):
    """
    双色球玄学维度对抗验证（P19）

    训练集(2020-2023) vs 测试集(2024-2025)回测对比，
    验证各维度是否过拟合。如果测试集提升率大幅衰减→过拟合。
    - **mode**: 旺行判定逻辑，默认day_zhi（v4.1推荐模式）
    - **birthday**: 出生日期（YYYY-MM-DD），传入后额外验证出生维度

    返回：训练集/测试集各维度命中率+提升值+衰减率+过拟合判定
    """
    if not _SSQ_HISTORY:
        raise HTTPException(status_code=503, detail="历史数据未加载")

    # 按日期拆分训练集和测试集
    train_data = [r for r in _SSQ_HISTORY if r.get("date") and "2020" <= r["date"][:4] <= "2023"]
    test_data = [r for r in _SSQ_HISTORY if r.get("date") and "2024" <= r["date"][:4] <= "2025"]

    if not train_data or not test_data:
        raise HTTPException(status_code=504, detail="训练集或测试集数据不足")

    # 分别跑回测
    train_result = await _run_backtest(train_data, len(train_data), mode, birthday)
    test_result = await _run_backtest(test_data, len(test_data), mode, birthday)

    train_details = {d["dimension"]: d for d in train_result["details"]}
    test_details = {d["dimension"]: d for d in test_result["details"]}

    # 对比分析
    comparison = []
    lines = [
        "【双色球玄学维度对抗验证（P19）】",
        "",
        f"训练集：2020-2023（{len(train_data)}期）",
        f"测试集：2024-2025（{len(test_data)}期）",
        f"旺行模式：{mode}",
        "",
        "对比逻辑：训练集提升率 → 测试集提升率 → 衰减率 → 过拟合判定",
        "衰减率 = (训练提升 - 测试提升) / |训练提升| × 100%",
        "判定标准：衰减<30%→✅稳健 | 30-60%→⚠️衰减 | 60-100%→❌严重衰减 | >100%→💀过拟合",
        "",
    ]

    lines.append(f"{'维度':<12} {'训练提升':>8} {'测试提升':>8} {'衰减率':>8} {'判定'}")
    lines.append("-" * 65)

    for dim_name in train_details:
        if dim_name not in test_details:
            continue
        t = train_details[dim_name]
        v = test_details[dim_name]

        t_lift = t["red_lift"]
        v_lift = v["red_lift"]

        # 衰减率计算（仅对训练集有正向信号的维度有意义）
        if abs(t_lift) > 0.5:
            decay_rate = round((t_lift - v_lift) / abs(t_lift) * 100, 1)
        else:
            decay_rate = 0.0

        # 过拟合判定（核心逻辑：训练集正信号→测试集是否保持）
        if abs(t_lift) <= 0.5 and abs(v_lift) <= 0.5:
            verdict = "➖无信号"
        elif t_lift > 0.5:
            # 训练集有正信号 → 检查测试集是否保持
            if v_lift > 2:
                verdict = "✅稳健(测试集仍有效)"
            elif v_lift > 0.5:
                if decay_rate < 60:
                    verdict = "✅稳健"
                else:
                    verdict = "⚠️衰减"
            elif v_lift > -0.5:
                verdict = "❌严重衰减"
            elif t_lift * v_lift < 0:
                verdict = "💀过拟合"
            else:
                verdict = "💀过拟合"
        elif t_lift < -0.5:
            # 训练集负信号 → 测试集方向变化
            if v_lift > 0.5:
                verdict = "🔄反转(负面→正面)"  # 好事！
            elif abs(v_lift) < abs(t_lift):
                verdict = "🔄减弱(负面减轻)"  # 好事！
            else:
                verdict = "❌持续负面"
        else:
            verdict = "➖无信号"

        comparison.append({
            "dimension": dim_name,
            "train_red_lift": t_lift,
            "test_red_lift": v_lift,
            "decay_rate": decay_rate,
            "train_verdict": t["verdict"],
            "test_verdict": v["verdict"],
            "overfit_verdict": verdict,
        })

        lines.append(
            f"{dim_name:<12} {t_lift:>+7.2f}% {v_lift:>+7.2f}% {decay_rate:>+7.1f}% {verdict}"
        )

    # 汇总统计
    robust_count = sum(1 for c in comparison if "稳健" in c["overfit_verdict"])
    decay_count = sum(1 for c in comparison if c["overfit_verdict"] == "⚠️衰减")
    severe_count = sum(1 for c in comparison if c["overfit_verdict"] in ("❌严重衰减", "💀过拟合"))
    neutral_count = sum(1 for c in comparison if c["overfit_verdict"] == "➖无信号")
    reverse_count = sum(1 for c in comparison if "反转" in c["overfit_verdict"] or "减弱" in c["overfit_verdict"])

    # 有效维度在测试集中的表现
    test_effective = [c for c in comparison if c["test_red_lift"] > 2]

    lines.append("")
    lines.append("━" * 65)
    lines.append(f"📊 汇总：稳健{robust_count} | 衰减{decay_count} | 严重/过拟合{severe_count} | 无信号{neutral_count} | 反转/减弱{reverse_count}")
    lines.append(f"📊 训练集有效维度(>2%)：{sum(1 for c in comparison if c['train_red_lift']>2)}个")
    lines.append(f"📊 测试集有效维度(>2%)：{len(test_effective)}个")
    if test_effective:
        lines.append(f"📊 测试集仍有效维度：{', '.join(c['dimension'] for c in test_effective)}")
    lines.append("")
    lines.append("💡 结论：")
    if len(test_effective) >= 3:
        lines.append(f"  ✅ {len(test_effective)}个维度在测试集仍有效，v4.1融合策略具备泛化能力")
    elif len(test_effective) >= 1:
        lines.append(f"  ⚠️ 仅{len(test_effective)}个维度在测试集仍有效，部分维度可能过拟合")
    else:
        lines.append(f"  ❌ 无维度在测试集仍有效，v4.1策略可能严重过拟合，需重新评估")
    lines.append("")
    lines.append("💡 使用建议：")
    lines.append("  1. 稳健维度→可信赖 | 衰减维度→降低权重 | 过拟合维度→考虑移除")
    lines.append("  2. 对抗验证是P17贝叶斯融合的先验可靠性依据")

    formatted_adversarial = "\n".join(lines)

    return {
        "train_periods": str(len(train_data)),
        "test_periods": str(len(test_data)),
        "mode": mode,
        "comparison": json.dumps(comparison, ensure_ascii=False),
        "formatted_adversarial": formatted_adversarial,
    }


@app.get("/ssq/pick", tags=["双色球历史数据"])
async def ssq_pick(date: str = "", mode: str = "auto", count: int = 5, birthday: str = ""):
    """
    双色球融合选号接口（玄学+统计）

    基于回测验证的有效维度+历史统计，融合生成推荐号码组合。
    - **date**: 公历日期（YYYY-MM-DD），默认今天
    - **mode**: 旺行判定逻辑，可选：
      - auto（默认）：自动推荐最优模式
      - day_gan：日柱天干五行
      - day_zhi：日柱地支五行
      - majority：六柱综合众数
    - **count**: 生成几注，默认5，最大10

    选号逻辑：
    1. 玄学有效维度（六柱干支/生我行/纳音五行/飞星，日月已移除-负面）缩范围 → 红球候选池
    2. 历史统计（频率/遗漏/冷热）二次筛选 → 加权排序
    3. 约束条件（和值/奇偶/大小/区间）优化组合
    4. 蓝球独立选号（玄学蓝球+统计蓝球融合）
    """
    import random
    from datetime import date as date_cls
    from lunarcalendar import Converter, Solar
    from collections import Counter

    if not _SSQ_HISTORY:
        raise HTTPException(status_code=503, detail="历史数据未加载")
    if mode not in ("day_gan", "day_zhi", "majority", "auto"):
        raise HTTPException(status_code=400, detail="mode参数错误，可选：day_gan / day_zhi / majority / auto")
    count = min(max(count, 1), 10)

    # 日期处理
    if not date:
        from datetime import datetime as _dt
        solar_date = _dt.now().date()
    else:
        try:
            parts = date.split('-')
            solar_date = date_cls(int(parts[0]), int(parts[1]), int(parts[2]))
        except:
            raise HTTPException(status_code=400, detail="日期格式错误")

    # ===== 第一步：玄学维度计算（复用/ganzhi逻辑）=====
    _TL = ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸']
    _DZ = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']
    base_date = date_cls(2000, 1, 7)

    y_offset = solar_date.year - 1984
    year_gan = _TL[y_offset % 10]
    year_zhi = _DZ[y_offset % 12]
    diff = (solar_date - base_date).days
    day_gan = _TL[diff % 10]
    day_zhi = _DZ[diff % 12]
    month_dz_map_p = {1:'丑', 2:'寅', 3:'卯', 4:'辰', 5:'巳', 6:'午',
                    7:'未', 8:'申', 9:'酉', 10:'戌', 11:'亥', 12:'子'}
    month_zhi = month_dz_map_p[solar_date.month]
    tg_start = {'甲':'丙','己':'丙','乙':'戊','庚':'戊','丙':'庚','辛':'庚',
                '丁':'壬','壬':'壬','戊':'甲','癸':'甲'}
    start_tg = tg_start[year_gan]
    start_idx = _TL.index(start_tg)
    m_dz_order = ['寅','卯','辰','巳','午','未','申','酉','戌','亥','子','丑']
    month_dz_idx = m_dz_order.index(month_zhi)
    month_gan = _TL[(start_idx + month_dz_idx) % 10]

    # 旺行
    six_wx = [
        _TIANGAN_MAP[year_gan]["wuxing"], _DIZHI_RED_MAP[year_zhi]["wuxing"],
        _TIANGAN_MAP[month_gan]["wuxing"], _DIZHI_RED_MAP[month_zhi]["wuxing"],
        _TIANGAN_MAP[day_gan]["wuxing"], _DIZHI_RED_MAP[day_zhi]["wuxing"],
    ]
    wx_counter = Counter(six_wx)

    # v3.5: auto模式自动推荐（基于六柱五行分布+回测结论）
    auto_reason = ""
    if mode == "auto":
        top_wx, top_count = wx_counter.most_common(1)[0]
        day_gan_wx = _TIANGAN_MAP[day_gan]["wuxing"]
        day_zhi_wx = _DIZHI_RED_MAP[day_zhi]["wuxing"]
        if top_count >= 4:
            # 众数五行出现4+次，majority更有效（生我行+7.86%）
            mode = "majority"
            auto_reason = f"auto→majority：六柱众数{top_wx}出现{top_count}次(≥4)，majority生我行+7.86%"
        elif day_zhi_wx != day_gan_wx and top_count >= 3:
            # 众数3次且日干支五行不同，day_zhi蓝球克我行+13.08%最有效
            mode = "day_zhi"
            auto_reason = f"auto→day_zhi：众数{top_wx}出现{top_count}次，日干({day_gan_wx})≠日支({day_zhi_wx})，day_zhi蓝球克我行+13.08%"
        else:
            # 默认day_gan，红球生我行+6.57%稳定有效
            mode = "day_gan"
            auto_reason = f"auto→day_gan：六柱分散(众数{top_wx}仅{top_count}次)，day_gan生我行+6.57%稳定"

    if mode == "day_zhi":
        day_wuxing = _DIZHI_RED_MAP[day_zhi]["wuxing"]
    elif mode == "majority":
        day_wuxing = wx_counter.most_common(1)[0][0]
    else:
        day_wuxing = _TIANGAN_MAP[day_gan]["wuxing"]

    shengke = _get_shengke_info(day_wuxing)

    # 玄学红球候选池（基于回测有效维度，日月已移除-负面）
    # 自适应权重配置v3.5（基于2144期全量回测，改此处即全局生效）
    # 按模式分组：day_gan用生我行，day_zhi用克我行+我生行·泄
    BACKTEST_WEIGHTS = {
        "day_gan": {
            "六柱干支": 3,    # 提升+15.05%，最有效维度
            "生我行":   2,    # 提升+6.57%
            "纳音五行": 2,    # 提升+5.53%
            "飞星":     2,    # 红球+0.73%中性偏正，提升覆盖
            "旺行":     0,    # 红球-5.24%负面，降权至0
            "克我行":   0,    # 红球-0.73%中性偏负
        },
        "day_zhi": {
            "六柱干支": 3,    # 提升+15.05%，最有效维度
            "我生行·泄": 2,   # 提升+5.28%
            "纳音五行": 2,    # 提升+5.53%
            "克我行":   2,    # 红球+3.54%✅，蓝球+13.08%✅ 提升×1→×2
            "飞星":     2,    # 红球+0.73%中性偏正，提升覆盖
            "旺行":     0,    # 红球+0.44%中性，蓝球+11.53%有效但红球弱
            "生我行":   0,    # 红球-2.21%负面，蓝球-23.55%大幅负面
        },
        "majority": {
            "六柱干支": 3,    # 提升+15.05%，最有效维度
            "生我行":   2,    # 提升+7.86%
            "纳音五行": 2,    # 提升+5.53%
            "飞星":     2,    # 红球+0.73%中性偏正，提升覆盖
            "旺行":     0,    # 红球-3.43%负面
            "我克行":   0,    # 红球-3.76%负面
        },
    }
    _weights = BACKTEST_WEIGHTS.get(mode, BACKTEST_WEIGHTS["day_gan"])
    _weight_str = " / ".join(f"{k}×{v}" for k,v in _weights.items() if v > 0)

    # 蓝球独立权重v3.5（基于2144期全量回测）
    # 红球用_weights，蓝球用_weights_blue
    BACKTEST_WEIGHTS_BLUE = {
        "day_gan": {
            "六柱干支": 2,    # 蓝球+1.41%弱有效
            "生我行":   0,    # 蓝球-5.62%负面！降权至0
            "纳音五行": 0,    # 蓝球-0.97%中性偏负
            "飞星":     1,    # 蓝球+0.73%中性
            "旺行":     1,    # 蓝球+8.43%有效！
            "克我行":   1,    # 蓝球+6.88%有效！v3.5新增（红球-0.73%但蓝球有效）
            "我生行·泄": 0,   # 蓝球无数据
        },
        "day_zhi": {
            "六柱干支": 1,    # 蓝球+1.41%弱有效
            "生我行":   0,    # 蓝球-23.55%大幅负面！必须降权
            "纳音五行": 0,    # 蓝球-0.97%中性偏负
            "克我行":   3,    # 蓝球+13.08%最有效！最高权重
            "我生行·泄": 0,   # 蓝球无数据
            "飞星":     1,    # 蓝球+0.73%中性
            "旺行":     2,    # 蓝球+11.53%有效！
        },
        "majority": {
            "六柱干支": 2,    # 蓝球+1.41%弱有效
            "生我行":   0,    # 蓝球-6.4%负面
            "纳音五行": 0,    # 蓝球-0.97%中性偏负
            "飞星":     1,    # 蓝球+0.73%中性
            "旺行":     1,    # 蓝球+3.78%弱有效
            "我克行":   0,    # 蓝球无数据
        },
    }
    _weights_blue = BACKTEST_WEIGHTS_BLUE.get(mode, BACKTEST_WEIGHTS_BLUE["day_gan"])

    xuanxue_red_score = {}
    xuanxue_blue_score = {}

    # 有效维度1：六柱干支（权重由BACKTEST_WEIGHTS配置）
    for tg, dz in [(year_gan, year_zhi), (month_gan, month_zhi), (day_gan, day_zhi)]:
        for n in _TIANGAN_MAP[tg]["red_balls"]:
            xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights["六柱干支"]
        for n in _DIZHI_RED_MAP[dz]["red_balls"]:
            xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights["六柱干支"]
        b = _DIZHI_BLUE_MAP[dz]
        xuanxue_blue_score[b] = xuanxue_blue_score.get(b, 0) + _weights_blue["六柱干支"]

    # 有效维度2：生我行（权重由_weights配置，day_zhi模式下降权至0）
    for n in _WUXING_MAP[shengke["生我行"]]["red_balls"]:
        xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights["生我行"]
    for n in _WUXING_MAP[shengke["生我行"]]["blue_balls"]:
        xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _weights_blue["生我行"]

    # 有效维度2b：克我行（day_zhi模式红球+3.54%，蓝球+13.08%显著有效）
    for n in _WUXING_MAP[shengke["克我行"]]["red_balls"]:
        xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights.get("克我行", 0)
    for n in _WUXING_MAP[shengke["克我行"]]["blue_balls"]:
        xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _weights_blue.get("克我行", 0)

    # 有效维度2c：我生行·泄（day_zhi模式红球+5.28%有效）
    for n in _WUXING_MAP[shengke["我生行(泄)"]]["red_balls"]:
        xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights.get("我生行·泄", 0)
    for n in _WUXING_MAP[shengke["我生行(泄)"]]["blue_balls"]:
        xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _weights_blue.get("我生行·泄", 0)

    # 有效维度3：纳音五行（权重由BACKTEST_WEIGHTS配置）
    day_ganzhi = day_gan + day_zhi
    day_nayin = _NAYIN_MAP.get(day_ganzhi, "")
    nayin_wuxing = _NAYIN_WUXING.get(day_nayin, "")
    if nayin_wuxing:
        for n in _WUXING_MAP[nayin_wuxing]["red_balls"]:
            xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights["纳音五行"]
        for n in _WUXING_MAP[nayin_wuxing]["blue_balls"]:
            xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _weights_blue["纳音五行"]

    # 中性维度：旺行（权重由BACKTEST_WEIGHTS配置，当前降权至0）
    for n in _WUXING_MAP[shengke["旺行"]]["red_balls"]:
        xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights["旺行"]
    for n in _WUXING_MAP[shengke["旺行"]]["blue_balls"]:
        xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _weights_blue["旺行"]

    # 中性维度：飞星方位（权重由BACKTEST_WEIGHTS配置）
    bagua = _DIZHI_BAGUA_MAP.get(day_zhi, {})
    if bagua:
        for n in bagua["red_balls"]:
            xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _weights["飞星"]
        for n in bagua["blue_balls"]:
            xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _weights_blue["飞星"]

    # ===== v5.0 P16: 二十八宿宇宙维度 =====
    zhixiu = _get_zhixiu(solar_date)
    qiyao = _get_qiyao(solar_date)
    _XIU_WEIGHT = 1  # 二十八宿权重（初始1，待回测调整）
    _YAO_WEIGHT = 1  # 七曜权重（初始1，待回测调整）

    # 值宿→五行→号码映射
    xiu_wuxing = zhixiu["wuxing"]
    for n in _WUXING_MAP[xiu_wuxing]["red_balls"]:
        xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _XIU_WEIGHT
    for n in _WUXING_MAP[xiu_wuxing]["blue_balls"]:
        xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _XIU_WEIGHT

    # 七曜→五行→号码映射
    yao_wuxing = qiyao["wuxing"]
    for n in _WUXING_MAP[yao_wuxing]["red_balls"]:
        xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + _YAO_WEIGHT
    for n in _WUXING_MAP[yao_wuxing]["blue_balls"]:
        xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + _YAO_WEIGHT

    # ===== v3.5 出生维度（可选，权重基于回测） =====
    b_day_gan_p = b_day_zhi_p = b_day_wuxing_p = b_shengke_p = None
    birth_weight_str = ""
    if birthday:
        try:
            b_parts = birthday.split('-')
            b_date_p = date_cls(int(b_parts[0]), int(b_parts[1]), int(b_parts[2]))
            b_diff_p = (b_date_p - base_date).days
            b_day_gan_p = _TL[b_diff_p % 10]
            b_day_zhi_p = _DZ[b_diff_p % 12]
            b_day_wuxing_p = _TIANGAN_MAP[b_day_gan_p]["wuxing"]
            b_shengke_p = _get_shengke_info(b_day_wuxing_p)
            # v3.5 出生维度权重（基于2144期回测：生我行+3.03%有效，克我行-3.25%负面）
            BIRTH_WEIGHTS = {"旺行": 1, "生我行": 2, "克我行": 0, "我生行·泄": 0}
            BIRTH_WEIGHTS_BLUE = {"旺行": 1, "克我行": 0, "生我行": 1, "我生行·泄": 0}
            # 红球
            for n in _WUXING_MAP[b_shengke_p["旺行"]]["red_balls"]:
                xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + BIRTH_WEIGHTS["旺行"]
            for n in _WUXING_MAP[b_shengke_p["生我行"]]["red_balls"]:
                xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + BIRTH_WEIGHTS["生我行"]
            for n in _WUXING_MAP[b_shengke_p["克我行"]]["red_balls"]:
                xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + BIRTH_WEIGHTS["克我行"]
            for n in _WUXING_MAP[b_shengke_p["我生行(泄)"]]["red_balls"]:
                xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + BIRTH_WEIGHTS["我生行·泄"]
            for n in _TIANGAN_MAP[b_day_gan_p]["red_balls"]:
                xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + 1
            for n in _DIZHI_RED_MAP[b_day_zhi_p]["red_balls"]:
                xuanxue_red_score[n] = xuanxue_red_score.get(n, 0) + 1
            # 蓝球
            for n in _WUXING_MAP[b_shengke_p["旺行"]]["blue_balls"]:
                xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + BIRTH_WEIGHTS_BLUE["旺行"]
            for n in _WUXING_MAP[b_shengke_p["生我行"]]["blue_balls"]:
                xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + BIRTH_WEIGHTS_BLUE["生我行"]
            for n in _WUXING_MAP[b_shengke_p["克我行"]]["blue_balls"]:
                xuanxue_blue_score[n] = xuanxue_blue_score.get(n, 0) + BIRTH_WEIGHTS_BLUE["克我行"]
            xuanxue_blue_score[_DIZHI_BLUE_MAP[b_day_zhi_p]] = xuanxue_blue_score.get(_DIZHI_BLUE_MAP[b_day_zhi_p], 0) + 1
            birth_weight_str = f"出生({b_day_gan_p}{b_day_zhi_p}日·{b_day_wuxing_p}行) 旺行×1/生我×2/克我×0/泄×0"
        except:
            pass  # birthday参数错误时静默忽略，不影响主流程

    # ===== 第二步：v4.1统计引擎升级（马尔可夫+衰减+关联规则+冷热周期+自适应权重）=====
    stat_periods = min(100, len(_SSQ_HISTORY))  # v4.0: 扩大到100期（原50期）
    stat_data = _SSQ_HISTORY[:stat_periods]

    # --- 引擎1: 衰减记忆加权频率（替代原等权频率） ---
    decay_freq_red = _decay_weighted_stats(stat_data, "red", decay=0.95)
    decay_freq_blue = _decay_weighted_stats(stat_data, "blue", decay=0.95)

    # --- 引擎1b: 衰减遗漏值（替代原等权遗漏） ---
    decay_miss_red = _decay_miss_with_weight(stat_data, "red", decay=0.95)
    decay_miss_blue = _decay_miss_with_weight(stat_data, "blue", decay=0.95)

    # --- 引擎2: 马尔可夫链转移预测 ---
    markov_trans_red = _markov_transition(stat_data, "red")
    markov_trans_blue = _markov_transition(stat_data, "blue")
    last_red = _SSQ_HISTORY[0]["red"]  # 最新一期红球
    last_blue = [_SSQ_HISTORY[0]["blue"]]  # 最新一期蓝球
    markov_pred_red = _markov_predict(markov_trans_red, last_red, "red")
    markov_pred_blue = _markov_predict(markov_trans_blue, last_blue, "blue")

    # --- 引擎3: 关联规则预测 ---
    assoc_rules_red = _association_rules(stat_data, "red", min_support=0.03, min_confidence=0.12)
    assoc_rules_blue = _association_rules(stat_data, "blue", min_support=0.02, min_confidence=0.08)
    assoc_pred_red = _association_predict(assoc_rules_red, last_red, "red")
    assoc_pred_blue = _association_predict(assoc_rules_blue, last_red, "blue")  # 蓝球用上一期红球作前件

    # --- v4.1 P8: 自适应引擎权重（基于近30期命中率动态调整） ---
    adaptive_weights = _adaptive_engine_weights(_SSQ_HISTORY, window=30)
    STAT_ENGINE_WEIGHTS = adaptive_weights
    _stat_weight_str = " / ".join(f"{k}×{int(v*100)}%" for k, v in STAT_ENGINE_WEIGHTS.items())

    stat_red_score = {}
    stat_blue_score = {}
    for n in range(1, 34):
        stat_red_score[n] = (
            decay_freq_red.get(n, 0) * STAT_ENGINE_WEIGHTS["decay_freq"] +
            decay_miss_red.get(n, 0) * STAT_ENGINE_WEIGHTS["decay_miss"] +
            markov_pred_red.get(n, 0) * STAT_ENGINE_WEIGHTS["markov"] +
            assoc_pred_red.get(n, 0) * STAT_ENGINE_WEIGHTS["association"]
        )
    for n in range(1, 17):
        stat_blue_score[n] = (
            decay_freq_blue.get(n, 0) * STAT_ENGINE_WEIGHTS["decay_freq"] +
            decay_miss_blue.get(n, 0) * STAT_ENGINE_WEIGHTS["decay_miss"] +
            markov_pred_blue.get(n, 0) * STAT_ENGINE_WEIGHTS["markov"] +
            assoc_pred_blue.get(n, 0) * STAT_ENGINE_WEIGHTS["association"]
        )

    # --- v4.1 P4: 冷热周期识别 ---
    hot_cold_red = _hot_cold_cycle(stat_data, "red", window=10)
    hot_cold_blue = _hot_cold_cycle(stat_data, "blue", window=10)

    # ===== 第三步：v5.0 贝叶斯融合评分（P17）=====
    # 核心公式：P(hi|d) ∝ P(d|hi) × P(hi)
    #   P(hi) = 玄学先验（基于回测有效维度的信念概率）
    #   P(d|hi) = 统计似然（基于4大统计引擎的观测概率）
    #   P(hi|d) = 后验概率（最终推荐得分）

    import math

    # --- P(hi): 玄学先验概率 ---
    # 将玄学得分转换为概率分布（Softmax归一化）
    # 温度参数：T越低→高得分号码概率越集中，T越高→分布越均匀
    _PRIOR_TEMP = 2.0  # 先验温度（玄学信念的集中度）

    # 红球先验
    _prior_red_raw = {}
    for n in range(1, 34):
        x = xuanxue_red_score.get(n, 0)
        _prior_red_raw[n] = math.exp(x / _PRIOR_TEMP)
    _prior_red_sum = sum(_prior_red_raw.values())
    prior_red = {n: v / _prior_red_sum for n, v in _prior_red_raw.items()}

    # 蓝球先验
    _prior_blue_raw = {}
    for n in range(1, 17):
        x = xuanxue_blue_score.get(n, 0)
        _prior_blue_raw[n] = math.exp(x / _PRIOR_TEMP)
    _prior_blue_sum = sum(_prior_blue_raw.values())
    prior_blue = {n: v / _prior_blue_sum for n, v in _prior_blue_raw.items()}

    # --- P(d|hi): 统计似然概率 ---
    # 将统计得分转换为似然函数（Softmax归一化）
    _LIKELIHOOD_TEMP = 1.5  # 似然温度（统计信号的集中度，更低=更集中）

    # 红球似然
    _like_red_raw = {}
    for n in range(1, 34):
        s = stat_red_score.get(n, 0)
        _like_red_raw[n] = math.exp(s / _LIKELIHOOD_TEMP)
    _like_red_sum = sum(_like_red_raw.values())
    likelihood_red = {n: v / _like_red_sum for n, v in _like_red_raw.items()}

    # 蓝球似然
    _like_blue_raw = {}
    for n in range(1, 17):
        s = stat_blue_score.get(n, 0)
        _like_blue_raw[n] = math.exp(s / _LIKELIHOOD_TEMP)
    _like_blue_sum = sum(_like_blue_raw.values())
    likelihood_blue = {n: v / _like_blue_sum for n, v in _like_blue_raw.items()}

    # --- P(hi|d): 后验概率 = 先验 × 似然 ---
    # 贝叶斯融合：后验 ∝ P(hi) × P(d|hi)
    # 用对数空间避免下溢：log P(hi|d) = log P(hi) + log P(d|hi) + const
    _COLD_FLOOR = 0.3  # 冷门号保底分（贝叶斯框架下缩小，避免0概率）

    # v4.1 P7: 玄学统计交叉验证（贝叶斯版：先验和似然一致的号码获得额外置信度）
    xuanxue_top9 = set(sorted(xuanxue_red_score, key=xuanxue_red_score.get, reverse=True)[:9])
    stat_top9 = set(sorted(stat_red_score, key=stat_red_score.get, reverse=True)[:9])
    cross_red = xuanxue_top9 & stat_top9

    xuanxue_blue_top3 = set(sorted(xuanxue_blue_score, key=xuanxue_blue_score.get, reverse=True)[:3])
    stat_blue_top3 = set(sorted(stat_blue_score, key=stat_blue_score.get, reverse=True)[:3])
    cross_blue = xuanxue_blue_top3 & stat_blue_top3

    _CROSS_BONUS = 0.3  # 贝叶斯版交叉加分（作为先验-似然一致的额外置信度）

    # 红球后验
    log_posterior_red = {}
    for n in range(1, 34):
        log_prior = math.log(max(prior_red[n], 1e-10))
        log_likelihood = math.log(max(likelihood_red[n], 1e-10))
        # P4: 冷热周期加分
        hc_score = hot_cold_red[n]["score"] * 1.0
        # P7: 交叉验证（先验-似然一致→额外置信度）
        cross = _CROSS_BONUS if n in cross_red else 0
        # 冷门号保底
        floor = _COLD_FLOOR if xuanxue_red_score.get(n, 0) == 0 and stat_red_score.get(n, 0) > 0 else 0
        # 后验 = log先验 + log似然 + 冷热周期 + 交叉 + 保底
        log_posterior_red[n] = log_prior + log_likelihood + hc_score + cross + floor

    # 蓝球后验
    log_posterior_blue = {}
    for n in range(1, 17):
        log_prior = math.log(max(prior_blue[n], 1e-10))
        log_likelihood = math.log(max(likelihood_blue[n], 1e-10))
        hc_score = hot_cold_blue[n]["score"] * 1.0
        cross = _CROSS_BONUS if n in cross_blue else 0
        floor = _COLD_FLOOR if xuanxue_blue_score.get(n, 0) == 0 and stat_blue_score.get(n, 0) > 0 else 0
        log_posterior_blue[n] = log_prior + log_likelihood + hc_score + cross + floor

    # 归一化后验概率（用于展示）
    _post_red_max = max(log_posterior_red.values())
    posterior_red = {n: math.exp(v - _post_red_max) for n, v in log_posterior_red.items()}
    _post_red_sum = sum(posterior_red.values())
    posterior_red = {n: v / _post_red_sum for n, v in posterior_red.items()}

    _post_blue_max = max(log_posterior_blue.values())
    posterior_blue = {n: math.exp(v - _post_blue_max) for n, v in log_posterior_blue.items()}
    _post_blue_sum = sum(posterior_blue.values())
    posterior_blue = {n: v / _post_blue_sum for n, v in posterior_blue.items()}

    # 用后验概率作为最终得分（乘以100便于展示和排序）
    final_red_score = {n: round(posterior_red[n] * 100, 1) for n in range(1, 34)}
    final_blue_score = {n: round(posterior_blue[n] * 100, 1) for n in range(1, 17)}

    # 红球排序
    sorted_red = sorted(final_red_score.items(), key=lambda x: (-x[1], x[0]))
    # 蓝球排序
    sorted_blue = sorted(final_blue_score.items(), key=lambda x: (-x[1], x[0]))

    # ===== 第四步：生成号码组合（统计约束优化）=====
    # 红球候选池：取TOP18
    red_pool = [n for n, _ in sorted_red[:18]]
    # 蓝球候选池：取TOP6
    blue_pool = [n for n, _ in sorted_blue[:6]]

    # 统计约束基准（近50期）
    _target_sum = (90, 120)       # 和值最优区间
    _good_odd = (2, 4)            # 奇数个数合理范围（3:3最优）
    _good_big = (2, 4)            # 大号个数合理范围（3:3最优）
    _min_zone = 1                 # 每区至少出1个

    random.seed(solar_date.toordinal())  # 日期固定种子

    def _combo_score(red6, blue):
        """组合质量评分：越高越均衡"""
        s = sum(red6)
        odd = sum(1 for n in red6 if n % 2 == 1)
        big = sum(1 for n in red6 if n >= 17)
        z1 = sum(1 for n in red6 if n <= 11)
        z2 = sum(1 for n in red6 if 12 <= n <= 22)
        z3 = sum(1 for n in red6 if n >= 23)
        score = 100.0

        # 和值约束（和值在90-120最优，偏离扣分）
        if s < _target_sum[0]:
            score -= (_target_sum[0] - s) * 2
        elif s > _target_sum[1]:
            score -= (s - _target_sum[1]) * 2

        # 奇偶约束（3:3最优=0扣分，4:2或2:4扣5分，5:1或1:5扣15分，6:0或0:6扣30分）
        if odd in (3,):  pass  # 3:3 完美
        elif odd in (2, 4):  score -= 5
        elif odd in (1, 5):  score -= 15
        else:  score -= 30

        # 大小约束（同理）
        if big in (3,):  pass
        elif big in (2, 4):  score -= 5
        elif big in (1, 5):  score -= 15
        else:  score -= 30

        # 区间约束（三区均出加分，某区0个扣分）
        zones = [z1, z2, z3]
        if all(z >= _min_zone for z in zones):
            score += 10  # 三区均出加分
        else:
            empty_zones = sum(1 for z in zones if z == 0)
            score -= empty_zones * 15

        # 玄学得分加成
        red_xuanxue = sum(final_red_score[n] for n in red6)
        score += red_xuanxue * 0.5

        # 蓝球得分加成
        score += final_blue_score.get(blue, 0) * 0.3

        return score

    # 生成大量候选组合，评分后取最优
    _candidates = []
    weights_red = [final_red_score[n] + 1 for n in red_pool]
    weights_blue = [final_blue_score[n] + 1 for n in blue_pool]

    for _ in range(count * 50):  # 生成50倍候选
        # 加权随机选6红
        chosen_red = []
        pool_nums = list(red_pool)
        pool_weights = list(weights_red)
        for _ in range(6):
            if not pool_nums:
                break
            s = sum(pool_weights)
            probs = [w / s for w in pool_weights] if s > 0 else [1/len(pool_weights)] * len(pool_weights)
            chosen = random.choices(pool_nums, weights=probs, k=1)[0]
            chosen_red.append(chosen)
            idx = pool_nums.index(chosen)
            pool_nums.pop(idx)
            pool_weights.pop(idx)

        chosen_red = sorted(chosen_red)

        # 蓝球加权随机
        chosen_blue = random.choices(blue_pool, weights=weights_blue, k=1)[0]

        # 计算组合质量分
        c_score = _combo_score(chosen_red, chosen_blue)

        sum_val = sum(chosen_red)
        odd = sum(1 for n in chosen_red if n % 2 == 1)
        big = sum(1 for n in chosen_red if n >= 17)
        z1 = sum(1 for n in chosen_red if n <= 11)
        z2 = sum(1 for n in chosen_red if 12 <= n <= 22)
        z3 = sum(1 for n in chosen_red if n >= 23)

        _candidates.append({
            "red": chosen_red,
            "blue": chosen_blue,
            "sum": sum_val,
            "odd_even": f"{odd}:{6-odd}",
            "big_small": f"{big}:{6-big}",
            "zones": f"{z1}:{z2}:{z3}",
            "_score": c_score,
        })

    # 去重（按红球集合去重）
    _seen = set()
    _unique = []
    for c in _candidates:
        key = tuple(c["red"])
        if key not in _seen:
            _seen.add(key)
            _unique.append(c)

    # 按质量分排序，取TOP count
    _unique.sort(key=lambda x: -x["_score"])
    combinations = _unique[:count]
    # 移除内部评分字段
    for c in combinations:
        c.pop("_score", None)

    # ===== 格式化输出 =====
    lines = [f"【双色球融合选号（{solar_date}）】", ""]
    if auto_reason:
        lines.append(f"💡 {auto_reason}")
    lines.append(f"玄学有效维度权重（自适应·{mode}模式）：{_weight_str}")
    lines.append(f"统计引擎v5.0自适应权重：{_stat_weight_str}")
    _blue_weight_str = " / ".join(f"{k}×{v}" for k,v in _weights_blue.items() if v > 0)
    if _blue_weight_str != _weight_str:
        lines.append(f"蓝球独立权重：{_blue_weight_str}")
    if birth_weight_str:
        lines.append(f"🎂出生维度权重：{birth_weight_str}")
    lines.append(f"🔮贝叶斯融合：先验P(hi)·玄学Softmax(T={_PRIOR_TEMP}) + 似然P(d|hi)·统计Softmax(T={_LIKELIHOOD_TEMP}) → 后验P(hi|d)")
    lines.append(f"🌌二十八宿：{zhixiu['xiang']}{zhixiu['desc']}·{zhixiu['wuxing']}行 | 七曜·{qiyao['desc']}·{qiyao['wuxing']}行")

    # v4.1 冷热周期摘要
    cold_to_warm = sorted([n for n in range(1,34) if hot_cold_red[n]["turn"] == "冷→温↑"])
    warm_to_hot = sorted([n for n in range(1,34) if hot_cold_red[n]["turn"] == "温→热↑"])
    hot_stable = sorted([n for n in range(1,34) if hot_cold_red[n]["turn"] == "热→"])
    hot_cooling = sorted([n for n in range(1,34) if hot_cold_red[n]["turn"] == "热→温↓"])
    lines.append(f"🔥冷热周期：冷转热{' '.join(f'{n:02d}' for n in cold_to_warm[:5])} | 温转热{' '.join(f'{n:02d}' for n in warm_to_hot[:5])} | 稳定热{' '.join(f'{n:02d}' for n in hot_stable[:5])} | 热转冷{' '.join(f'{n:02d}' for n in hot_cooling[:5])}")

    # v4.1 交叉验证号码
    cross_nums = sorted(cross_red)
    cross_blue_nums = sorted(cross_blue)
    lines.append(f"🔥交叉验证：红球{' '.join(f'{n:02d}' for n in cross_nums)} | 蓝球{' '.join(f'{n:02d}' for n in cross_blue_nums)}")

    # 红球TOP18带交叉标记
    red_pool_marks = []
    for n in red_pool:
        mark = "🔥交叉" if n in cross_red else ""
        red_pool_marks.append(f"{n:02d}({final_red_score[n]:.1f}){mark}")
    lines.append(f"红球候选TOP18：{', '.join(red_pool_marks)}")
    blue_pool_marks = []
    for n in blue_pool:
        bcross = "🔥" if n in cross_blue else ""
        blue_pool_marks.append(f"{n:02d}({final_blue_score[n]:.1f}){bcross}")
    lines.append(f"蓝球候选TOP6：{', '.join(blue_pool_marks)}")
    lines.append("")

    for i, combo in enumerate(combinations, 1):
        red_str = " ".join(f"{n:02d}" for n in combo["red"])
        lines.append(
            f"第{i}注：{red_str} + {combo['blue']:02d}  "
            f"和值={combo['sum']} 奇偶={combo['odd_even']} 大小={combo['big_small']} 区间={combo['zones']}"
        )

    formatted_pick = "\n".join(lines)

    return {
        "date": str(solar_date),
        "mode": mode,
        "formatted_pick": formatted_pick,
        "combinations": json.dumps(combinations, ensure_ascii=False),
        "red_pool_top18": json.dumps(red_pool, ensure_ascii=False),
        "blue_pool_top6": json.dumps(blue_pool, ensure_ascii=False),
    }


if __name__ == "__main__":
    import uvicorn
    port = int(_os.environ.get("PORT", "8000"))
    print(f"===== STARTUP: Using PORT={port}, RAILWAY_ENV={_os.environ.get('RAILWAY_ENVIRONMENT', 'NOT SET')} =====", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
