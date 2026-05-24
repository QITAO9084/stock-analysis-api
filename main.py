from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")

def now_cn():
    """返回北京时间 ISO 字符串"""
    return datetime.now(CN_TZ).isoformat()
from typing import Optional
import time
import json
import threading

app = FastAPI(
    title="Stock Analysis API",
    description="股票/加密货币分析API - V5.18.2（信号源统一+KDJ/RSI标签修正+评级对齐）",
    version="5.18.1",
    servers=[{"url": "https://stock-analysis-api-n741.onrender.com", "description": "Render部署"}],
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== V5.12.1 Coze URL 修正中间件 =====
# 根因：Coze 忽略 OpenAPI servers 字段，直接用 spec URL 拼接工具路径
# 结果：/openapi.json/stock/analyze2 而非 /stock/analyze2 → 404
# 解决：直接修改 scope["path"] 绕过请求重建问题
from starlette.types import Scope
@app.middleware("http")
async def coze_url_fix_middleware(request: Request, call_next):
    path = request.scope.get("path", "")
    if path.startswith("/openapi.json/"):
        new_path = path.replace("/openapi.json", "", 1)
        request.scope["path"] = new_path
        # 同步 raw_path（若存在）
        if "raw_path" in request.scope:
            request.scope["raw_path"] = new_path.encode("latin-1")
    response = await call_next(request)
    return response

# ===== V5.6 优雅降级：将 404/500 错误转为正常 200 响应，避免 Agent 进入故障模式 =====
# 核心思路：Agent 看到 HTTP 错误 → 本能"解释+替代" → 兜底规则失效
# 解决方案：API 永不返回错误，工具永远"成功"，Agent 只需按数据输出

ERROR_FALLBACK_MESSAGE = "抱歉，数据获取暂时异常，请稍后再试。"

@app.exception_handler(HTTPException)
async def graceful_error_handler(request: Request, exc: HTTPException):
    """404/500 错误 → 正常 200 响应（防止 Agent 故障模式）"""
    if exc.status_code in (404, 500):
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "signal": "error",
                "message": ERROR_FALLBACK_MESSAGE,
                "formatted_report": ERROR_FALLBACK_MESSAGE,
            }
        )
    # 400 等参数错误正常返回
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )

# 未捕获异常也优雅降级
@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={
            "status": "error",
            "signal": "error",
            "message": ERROR_FALLBACK_MESSAGE,
            "formatted_report": ERROR_FALLBACK_MESSAGE,
        }
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

def calculate_adx(data, period=14):
    """计算 ADX 趋势强度指标

    返回: {"adx": float, "plus_di": float, "minus_di": float, "trend": str}
    trend: "strong_bull" (ADX>25 + +DI>-DI), "strong_bear" (ADX>25 + -DI>+DI),
           "weak_bull", "weak_bear", "ranging" (ADX<20 无趋势)
    """
    high = data['High']
    low = data['Low']
    close = data['Close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=data.index)
    minus_dm = pd.Series(0.0, index=data.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smooth with Wilder's method (EMA with alpha=1/period)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

    # Directional Index (DX) and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    adx_val = round(adx.iloc[-1], 2) if not pd.isna(adx.iloc[-1]) else 20
    pdi_val = round(plus_di.iloc[-1], 2) if not pd.isna(plus_di.iloc[-1]) else 25
    mdi_val = round(minus_di.iloc[-1], 2) if not pd.isna(minus_di.iloc[-1]) else 25

    # 趋势判断
    if adx_val >= 25:
        if pdi_val > mdi_val:
            trend = "strong_bull"
        else:
            trend = "strong_bear"
    elif adx_val >= 20:
        if pdi_val > mdi_val:
            trend = "weak_bull"
        else:
            trend = "weak_bear"
    else:
        trend = "ranging"

    return {
        "adx": adx_val,
        "plus_di": pdi_val,
        "minus_di": mdi_val,
        "trend": trend
    }

def detect_rsi_divergence(data):
    """检测 RSI 背离信号

    顶背离：价格创新高，RSI 未创新高 → 看空反转信号
    底背离：价格创新低，RSI 未创新低 → 看涨反转信号

    返回: {"type": str, "description": str}
    type: "bearish_divergence" / "bullish_divergence" / "none"
    """
    lookback = 20
    if len(data) < lookback + 14:
        return {"type": "none", "description": ""}

    # 计算 RSI 完整序列
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))

    # 取最近 lookback 天的价格和 RSI
    recent_close = data['Close'].tail(lookback)
    recent_rsi = rsi_series.tail(lookback)

    # 找峰值和谷值
    price_high = recent_close.max()
    price_low = recent_close.min()

    # 顶背离：价格创近期新高，RSI未创同期新高
    price_high_idx = recent_close.idxmax()
    rsi_high_in_window = recent_rsi.max()

    # 价格创新高（当前价接近最高点）且 RSI 比同期高点低 5+ 个点
    current_close = data['Close'].iloc[-1]
    current_rsi = rsi_series.iloc[-1] if not pd.isna(rsi_series.iloc[-1]) else 50

    if current_close >= price_high * 0.97:  # 当前价格接近或等于近期新高
        if current_rsi < rsi_high_in_window - 5:  # RSI 明显低于同期 RSI 高点
            return {
                "type": "bearish_divergence",
                "description": f"顶背离：价格接近{lookback}日高点({round(price_high,2)})，但RSI({round(current_rsi,1)})远低于同期RSI高点({round(rsi_high_in_window,1)})，上涨动能衰竭"
            }

    # 底背离：价格创近期新低，RSI未创同期新低
    price_low_idx = recent_close.idxmin()
    rsi_low_in_window = recent_rsi.min()

    if current_close <= price_low * 1.03:  # 当前价格接近或等于近期新低
        if current_rsi > rsi_low_in_window + 5:  # RSI 明显高于同期 RSI 低点
            return {
                "type": "bullish_divergence",
                "description": f"底背离：价格接近{lookback}日低点({round(price_low,2)})，但RSI({round(current_rsi,1)})远高于同期RSI低点({round(rsi_low_in_window,1)})，下跌动能衰竭"
            }

    return {"type": "none", "description": ""}


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
    adx_data = calculate_adx(data)
    rsi_divergence = detect_rsi_divergence(data)
    # V5.18: 资金流向指标
    obv_data = calculate_obv(data)
    mfi_data = calculate_mfi(data)
    cmf_data = calculate_cmf(data)
    vwap_data = calculate_vwap(data)

    buy_score = 0
    sell_score = 0
    buy_signals = []
    sell_signals = []

    # ---- 第一阶段：独立指标评分 ----

    # 1. RSI评分（V5.14: 更细粒度极值加权）
    if rsi < 20:
        buy_signals.append(f"RSI极度超卖（{round(rsi,1)}），强烈反弹信号")
        buy_score += 4
    elif rsi < 30:
        buy_signals.append(f"RSI超卖（{round(rsi,1)}），可能反弹")
        buy_score += 2
    elif rsi < 45:
        buy_signals.append(f"RSI偏低（{round(rsi,1)}），可考虑建仓")
        buy_score += 1

    if rsi > 80:
        sell_signals.append(f"RSI极度超买（{round(rsi,1)}），强烈回调信号")
        sell_score += 4
    elif rsi > 70:
        sell_signals.append(f"RSI超买（{round(rsi,1)}），注意风险")
        sell_score += 2
    elif rsi > 60:
        sell_signals.append(f"RSI偏高（{round(rsi,1)}），可考虑减仓")
        sell_score += 1

    # RSI趋势：超买但正在快速回落（比单纯超买更危险）
    if rsi > 70 and rsi_delta < -2:
        sell_signals.append(f"RSI超买且快速下降（{round(rsi,1)}→变化{rsi_delta}），注意回调")
        sell_score += 2
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

    # 7. ADX趋势强度评分（P0新增！）
    adx_trend = adx_data["trend"]
    if adx_trend == "strong_bull":
        buy_signals.append(f"ADX={adx_data['adx']}，强势多头趋势（+DI={adx_data['plus_di']} > -DI={adx_data['minus_di']}）")
        buy_score += 3
    elif adx_trend == "strong_bear":
        sell_signals.append(f"ADX={adx_data['adx']}，强势空头趋势（-DI={adx_data['minus_di']} > +DI={adx_data['plus_di']}）")
        sell_score += 3
    elif adx_trend == "weak_bull":
        buy_signals.append(f"ADX={adx_data['adx']}，趋势偏多但强度不足")
        buy_score += 1
    elif adx_trend == "weak_bear":
        sell_signals.append(f"ADX={adx_data['adx']}，趋势偏空但强度不足")
        sell_score += 1
    elif adx_trend == "ranging":
        # 震荡市：不直接加分，但降低其他指标的权重（在第三阶段处理）
        buy_signals.append(f"ADX={adx_data['adx']}，市场处于震荡格局，趋势信号可信度降低")

    # 8. RSI背离检测（P0新增！——最可靠的反转信号）
    if rsi_divergence["type"] == "bearish_divergence":
        sell_signals.insert(0, f"⚠️ {rsi_divergence['description']}")
        sell_score += 4  # 顶背离是最强卖出信号之一
    elif rsi_divergence["type"] == "bullish_divergence":
        buy_signals.insert(0, f"⚠️ {rsi_divergence['description']}")
        buy_score += 4  # 底背离是最强买入信号之一

    # V5.18: 9-12. 资金流向评分（OBV背离/MFI/CMF/VWAP）
    # OBV背离
    if obv_data["divergence"] == "bullish":
        buy_signals.insert(0, obv_data["desc"])
        buy_score += 3  # 底背离=资金暗中吸筹
    elif obv_data["divergence"] == "bearish":
        sell_signals.insert(0, obv_data["desc"])
        sell_score += 3  # 顶背离=资金暗中撤退
    elif obv_data["desc"]:
        buy_signals.append(obv_data["desc"])
        buy_score += 1

    # MFI超买超卖
    if mfi_data["zone"] == "超卖":
        buy_signals.append(f"MFI={mfi_data['mfi']}，资金流量超卖，短期可能回流")
        buy_score += 2
    elif mfi_data["zone"] == "超买":
        sell_signals.append(f"MFI={mfi_data['mfi']}，资金流量超买，短期可能出逃")
        sell_score += 2

    # CMF 资金流向
    if cmf_data["cmf"] > 0.15:
        buy_signals.append(f"CMF={cmf_data['cmf']}，资金持续流入，买方主导")
        buy_score += 2
    elif cmf_data["cmf"] < -0.15:
        sell_signals.append(f"CMF={cmf_data['cmf']}，资金持续流出，卖方主导")
        sell_score += 2

    # VWAP 位置
    vwap_pos_text = "上方" if vwap_data["position"] == "above" else "下方"
    if vwap_data["position"] == "above" and vwap_data["distance"] > 1:
        buy_signals.append(f"价格高于VWAP（+{vwap_data['distance']}%），买方主导")
        buy_score += 1
    elif vwap_data["position"] == "below" and abs(vwap_data["distance"]) > 1:
        sell_signals.append(f"价格低于VWAP（{vwap_data['distance']}%），卖方主导")
        sell_score += 1

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
            "adx": adx_data,
            "rsi_divergence": rsi_divergence,
        },
        "volume_signal": vol_status,
        "volume_ratio": vol_ratio,
        "support_level": round(recent_low, 2),
        "resistance_level": round(recent_high, 2),
        "trend_direction": "bullish" if (buy_score > sell_score) else ("bearish" if (sell_score > buy_score) else "neutral"),
        "timestamp": now_cn()
    }

@app.get("/")
def read_root():
    """API健康检查"""
    return {
        "status": "ok",
        "message": "Stock Analysis API is running",
        "version": "5.12.1"
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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 自动检测
    detect_market = market if market else "us"

    symbol, market = normalize_stock_symbol(symbol, detect_market)

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
            "timestamp": now_cn()
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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 自动检测
    detect_market = market if market else "us"

    symbol, market = normalize_stock_symbol(symbol, detect_market)

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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 自动检测
    detect_market = market if market else "us"

    symbol, market = normalize_stock_symbol(symbol, detect_market)

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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 自动检测
    detect_market = market if market else "us"

    symbol, market = normalize_stock_symbol(symbol, detect_market)

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
            "analysis_time": now_cn(),
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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 自动检测
    detect_market = market if market else "us"

    symbol, market = normalize_stock_symbol(symbol, detect_market)

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

        # V5.15: K线形态 + 成交量背离（用于报告展示）
        candle_patterns = detect_candlestick_patterns(data)
        vol_divergences = detect_volume_divergence(data['Close'], data['Volume'])
        cp_text = "；".join([f"{cp['pattern']}({cp['desc']})" for cp in candle_patterns]) if candle_patterns else "无特殊形态"
        vd_text = "；".join([f"{'🟢' if vd['type']=='bullish' else '🔴'}{vd['desc']}" for vd in vol_divergences]) if vol_divergences else "未检测到价量背离"

        # V5.18: 资金流向指标（OBV/MFI/CMF/VWAP）
        obv_data = calculate_obv(data)
        mfi_data = calculate_mfi(data)
        cmf_data = calculate_cmf(data)
        vwap_data = calculate_vwap(data)

        # V5.13: 信号准确率回测
        accuracy_data = compute_signal_accuracy(data)

        # V5.14: 多周期共振 — 周线和月线趋势
        weekly_trend = None
        monthly_trend = None
        market_trend = None
        try:
            _rate_limit_wait()
            ticker_w = yf.Ticker(symbol)
            data_weekly = ticker_w.history(period="2y", interval="1wk")
            if not data_weekly.empty:
                weekly_trend = get_period_trend(data_weekly)
        except Exception:
            pass
        try:
            _rate_limit_wait()
            ticker_m = yf.Ticker(symbol)
            data_monthly = ticker_m.history(period="5y", interval="1mo")
            if not data_monthly.empty:
                monthly_trend = get_period_trend(data_monthly)
        except Exception:
            pass
        # V5.14: 大盘环境因子
        try:
            market_trend = get_market_trend(str(market))
        except Exception:
            pass

        # V5.17.6: 基本面摘要
        fundamentals = {}
        try:
            pe = info.get("trailingPE")
            fwd_pe = info.get("forwardPE")
            mcap = info.get("marketCap")
            fundamentals["pe"] = round(pe, 1) if pe else None
            fundamentals["forward_pe"] = round(fwd_pe, 1) if fwd_pe else None
            fundamentals["market_cap"] = mcap
            fundamentals["sector"] = info.get("sector", "") or ""
            fundamentals["industry"] = info.get("industry", "") or ""
            fundamentals["beta"] = round(info.get("beta", 0), 2) if info.get("beta") else None
            fundamentals["52w_high"] = info.get("fiftyTwoWeekHigh")
            fundamentals["52w_low"] = info.get("fiftyTwoWeekLow")
            div_yield_raw = info.get("dividendYield")
            div_rate = info.get("dividend_rate")  # 每股年股息
            current_price = data['Close'].iloc[-1] if not data.empty else None
            # V5.18.1: 股息率计算优先用 div_rate/price 自己算，避免 yfinance 返回格式不一致
            if div_rate and current_price and current_price > 0:
                fundamentals["dividend_yield"] = round(div_rate / current_price * 100, 2)
            elif div_yield_raw is not None:
                # yfinance 格式混乱：有时是小数(0.0035)，有时是百分比(0.35)，有时是乱值(35.0)
                # 安全策略：如果值<1 说明是小数格式，需要*100；否则已是百分比
                # 但 35% 显然不合理（苹果历史最高~2.5%），做合理性校验
                if div_yield_raw < 1:
                    computed = round(div_yield_raw * 100, 2)
                    fundamentals["dividend_yield"] = computed if computed <= 15 else round(div_yield_raw, 2)
                else:
                    fundamentals["dividend_yield"] = round(div_yield_raw, 2) if div_yield_raw <= 15 else None
            else:
                fundamentals["dividend_yield"] = None
        except Exception:
            pass

        # V5.11: API层预渲染报告，Agent只做管道转发
        formatted_report, score_data = build_formatted_report(
            name=str(info.get("longName", symbol)),
            symbol=symbol,
            market=str(market),
            currency=str(info.get("currency", "USD")),
            current_price=current_price,
            change_percent=change_percent,
            # V5.18.2: 改用 detect_trade_points 的 trade_point 作为统一信号源，
            # 避免 get_trading_signal 说"看多"但买卖点给出做空价格的两张皮问题
            signal=str(trade_points["trade_point"]),
            confidence=("强" if abs(trade_points["score"]) >= 6 else ("中等" if abs(trade_points["score"]) >= 3 else "弱")),
            trade_point=str(trade_points["trade_point"]),
            trade_point_cn=trade_point_cn.get(trade_points["trade_point"], "观望"),
            trade_score=trade_points["score"],
            buy_reasons_text=buy_reasons_text,
            sell_reasons_text=sell_reasons_text,
            entry_price=trade_points["entry_price"],
            stop_loss=trade_points["stop_loss"],
            take_profit=trade_points["take_profit"],
            rsi=round(indicators["rsi"], 2),
            rsi_prev=round(indicators["rsi_prev"], 2),
            rsi_delta=indicators["rsi_delta"],
            macd_val=round(indicators["macd"]["macd"], 4),
            macd_signal=round(indicators["macd"]["signal"], 4),
            macd_hist=round(indicators["macd"]["histogram"], 4),
            macd_cross="golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none"),
            kdj_k=round(indicators["kdj"]["k"], 2),
            kdj_d=round(indicators["kdj"]["d"], 2),
            kdj_j=round(indicators["kdj"]["j"], 2),
            adx=round(indicators["adx"]["adx"], 2),
            adx_trend=str(indicators["adx"]["trend"]),
            plus_di=round(indicators["adx"]["plus_di"], 2),
            minus_di=round(indicators["adx"]["minus_di"], 2),
            boll_upper=round(indicators["bollinger_bands"]["upper"], 2),
            boll_middle=round(indicators["bollinger_bands"]["middle"], 2),
            boll_lower=round(indicators["bollinger_bands"]["lower"], 2),
            ma5=round(indicators["ma5"], 2),
            ma10=round(indicators["ma10"], 2),
            ma20=round(indicators["ma20"], 2),
            ma50=round(indicators["ma50"], 2) if indicators["ma50"] else 0,
            volume_signal=str(signal_data["volume_signal"]),
            volume_ratio=signal_data["volume_ratio"],
            trend_direction=str(signal_data["trend_direction"]),
            support_level=signal_data["support_level"],
            resistance_level=signal_data["resistance_level"],
            kline_text="\n".join(kline_text_lines),
            rsi_div_type=str(indicators["rsi_divergence"]["type"]),
            rsi_div_desc=str(indicators["rsi_divergence"]["description"]),
            accuracy_data=accuracy_data,
            weekly_trend=weekly_trend,
            monthly_trend=monthly_trend,
            market_trend=market_trend,
            candle_patterns=candle_patterns,
            vol_divergences=vol_divergences,
            atr=trade_points.get("atr"),
            fundamentals=fundamentals,
            obv_data=obv_data,
            mfi=mfi_data["mfi"],
            cmf=cmf_data["cmf"],
            vwap_data=vwap_data,
        )

        return {
            # 基础信息
            "symbol": str(symbol),
            "name": str(info.get("longName", "N/A")),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": str(info.get("currency", "USD")),
            "market": str(market),
            "analysis_time": now_cn(),
            # V5.13: 预渲染报告（Agent直接输出，无需加工）
            "formatted_report": formatted_report,
            # V5.17.5: 10维评分明细（程序化消费）
            "total_score": score_data["total_score"],
            "score_detail": score_data["score_breakdown"],
            # V5.13: 信号准确率回测数据
            "signal_accuracy": accuracy_data.get("accuracy"),
            "signal_accuracy_days": accuracy_data.get("testable_days", 0),
            "signal_accuracy_consistent": accuracy_data.get("consistent_days", 0),
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
            # P0新增：ADX趋势强度 + RSI背离
            "adx": round(indicators["adx"]["adx"], 2),
            "adx_trend": str(indicators["adx"]["trend"]),
            "plus_di": round(indicators["adx"]["plus_di"], 2),
            "minus_di": round(indicators["adx"]["minus_di"], 2),
            "rsi_divergence_type": str(indicators["rsi_divergence"]["type"]),
            "rsi_divergence_desc": str(indicators["rsi_divergence"]["description"]),
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
            # V5.14: 多周期&大盘
            "weekly_trend": weekly_trend.get("label", "—") if weekly_trend else "—",
            "monthly_trend": monthly_trend.get("label", "—") if monthly_trend else "—",
            "market_index_trend": market_trend.get("label", "—") if market_trend else "—",
            # V5.15: K线形态 & 成交量背离
            "candle_patterns": cp_text,
            "volume_divergence": vd_text,
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
                    "adx": round(indicators["adx"]["adx"], 2),
                    "adx_trend": str(indicators["adx"]["trend"]),
                    "rsi_divergence_type": str(indicators["rsi_divergence"]["type"]),
                    "rsi_divergence_desc": str(indicators["rsi_divergence"]["description"]),
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
            # ADX最强趋势（P0新增）
            adx_sorted = sorted(valid_results, key=lambda x: x["adx"], reverse=True)
            summary["strongest_trend"] = {"symbol": adx_sorted[0]["symbol"], "adx": adx_sorted[0]["adx"], "adx_trend": adx_sorted[0]["adx_trend"]}
            # RSI背离股票（P0新增）
            divergence_stocks = [r["symbol"] for r in valid_results if r["rsi_divergence_type"] != "none"]
            if divergence_stocks:
                summary["divergence_warnings"] = divergence_stocks

        analysis_time = now_cn()
        return {
            "market": market,
            "total": len(results),
            "success": len(valid_results),
            "stocks_text": json.dumps(results, ensure_ascii=False),
            "summary_text": json.dumps(summary, ensure_ascii=False),
            "analysis_time": analysis_time,
            # V5.12: API层预渲染多股对比报告
            "formatted_report": build_compare_report(
                market=market,
                total=len(results),
                success=len(valid_results),
                stocks=results,
                summary=summary,
            ),
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
            "analysis_time": now_cn(),
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
            # P0新增：ADX趋势强度 + RSI背离
            "adx": round(indicators["adx"]["adx"], 2),
            "adx_trend": str(indicators["adx"]["trend"]),
            "plus_di": round(indicators["adx"]["plus_di"], 2),
            "minus_di": round(indicators["adx"]["minus_di"], 2),
            "rsi_divergence_type": str(indicators["rsi_divergence"]["type"]),
            "rsi_divergence_desc": str(indicators["rsi_divergence"]["description"]),
            # K线（文本格式）
            "kline_text": "\n".join(kline_text_lines),
            # V5.11: API层预渲染报告
            "formatted_report": build_crypto_report(
                name=str(info.get("shortName", coin_name)),
                symbol=str(coin_name),
                current_price=current_price,
                change_percent=change_percent,
                signal=str(signal_data["signal"]),
                confidence=str(signal_data["confidence"]),
                signals_text=signals_text,
                rsi=round(indicators["rsi"], 2),
                rsi_prev=round(indicators["rsi_prev"], 2),
                rsi_delta=indicators["rsi_delta"],
                macd_val=round(indicators["macd"]["macd"], 4),
                macd_signal_val=round(indicators["macd"]["signal"], 4),
                macd_hist=round(indicators["macd"]["histogram"], 4),
                macd_cross="golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none"),
                kdj_k=round(indicators["kdj"]["k"], 2),
                kdj_d=round(indicators["kdj"]["d"], 2),
                kdj_j=round(indicators["kdj"]["j"], 2),
                adx=round(indicators["adx"]["adx"], 2),
                adx_trend=str(indicators["adx"]["trend"]),
                plus_di=round(indicators["adx"]["plus_di"], 2),
                minus_di=round(indicators["adx"]["minus_di"], 2),
                boll_upper=round(indicators["bollinger_bands"]["upper"], 2),
                boll_middle=round(indicators["bollinger_bands"]["middle"], 2),
                boll_lower=round(indicators["bollinger_bands"]["lower"], 2),
                ma5=round(indicators["ma5"], 2),
                ma10=round(indicators["ma10"], 2),
                ma20=round(indicators["ma20"], 2),
                ma50=round(indicators["ma50"], 2) if indicators["ma50"] else 0,
                volume_signal=str(signal_data["volume_signal"]),
                volume_ratio=signal_data["volume_ratio"],
                support_level=signal_data["support_level"],
                resistance_level=signal_data["resistance_level"],
                kline_text="\n".join(kline_text_lines),
                rsi_div_type=str(indicators["rsi_divergence"]["type"]),
                rsi_div_desc=str(indicators["rsi_divergence"]["description"]),
            ),
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
            "analysis_time": now_cn(),
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
            # V5.11: API层预渲染报告
            "formatted_report": build_forex_report(
                pair=str(pair_upper),
                name=str(pair_name),
                current_price=current_price,
                change_percent=change_percent,
                signal=str(signal_data["signal"]),
                confidence=str(signal_data["confidence"]),
                signals_text=signals_text,
                rsi=round(indicators["rsi"], 2),
                rsi_prev=round(indicators["rsi_prev"], 2),
                rsi_delta=indicators["rsi_delta"],
                macd_val=round(indicators["macd"]["macd"], 4),
                macd_signal_val=round(indicators["macd"]["signal"], 4),
                macd_hist=round(indicators["macd"]["histogram"], 4),
                macd_cross="golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none"),
                kdj_k=round(indicators["kdj"]["k"], 2),
                kdj_d=round(indicators["kdj"]["d"], 2),
                kdj_j=round(indicators["kdj"]["j"], 2),
                adx=round(indicators["adx"]["adx"], 2),
                adx_trend=str(indicators["adx"]["trend"]),
                plus_di=round(indicators["adx"]["plus_di"], 2),
                minus_di=round(indicators["adx"]["minus_di"], 2),
                boll_upper=round(indicators["bollinger_bands"]["upper"], 4),
                boll_middle=round(indicators["bollinger_bands"]["middle"], 4),
                boll_lower=round(indicators["bollinger_bands"]["lower"], 4),
                ma5=round(indicators["ma5"], 4),
                ma10=round(indicators["ma10"], 4),
                ma20=round(indicators["ma20"], 4),
                ma50=round(indicators["ma50"], 4) if indicators["ma50"] else 0,
                volume_signal=str(signal_data["volume_signal"]),
                volume_ratio=signal_data["volume_ratio"],
                support_level=recent_low,
                resistance_level=recent_high,
                kline_text="\n".join(kline_text_lines),
                rsi_div_type=str(indicators["rsi_divergence"]["type"]),
                rsi_div_desc=str(indicators["rsi_divergence"]["description"]),
                volatility_20d=volatility_20d,
            ),
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
    adx_data = calculate_adx(data)                      # V5.9: ADX趋势强度
    divergence_data = detect_rsi_divergence(data)        # V5.9: RSI背离检测
    obv_data = calculate_obv(data)                       # V5.18: OBV资金流向
    mfi_data = calculate_mfi(data)                       # V5.18: MFI资金流量
    cmf_data = calculate_cmf(data)                       # V5.18: CMF蔡金资金流
    vwap_data = calculate_vwap(data)                     # V5.18: VWAP加权均价

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

    # ========== V5.9: ADX趋势强度 ==========
    # 7. ADX趋势方向（强趋势下顺趋势操作加分）
    adx_trend = adx_data.get("trend", "ranging")
    adx_val = adx_data.get("adx", 0)
    plus_di = adx_data.get("plus_di", 0)
    minus_di = adx_data.get("minus_di", 0)
    if adx_trend == "strong_bull":
        buy_reasons.append(f"ADX强势多头（ADX={round(adx_val,1)}，+DI={round(plus_di,1)}>-DI={round(minus_di,1)}），趋势确立")
        buy_count += 1
    elif adx_trend == "strong_bear":
        sell_reasons.append(f"ADX强势空头（ADX={round(adx_val,1)}，-DI={round(minus_di,1)}>+DI={round(plus_di,1)}），趋势确立")
        sell_count += 1
    elif adx_trend == "weak_bull":
        buy_reasons.append(f"ADX偏多（ADX={round(adx_val,1)}），趋势正在形成")
        buy_count += 0.5
    elif adx_trend == "weak_bear":
        sell_reasons.append(f"ADX偏空（ADX={round(adx_val,1)}），趋势正在形成")
        sell_count += 0.5
    # ranging (ADX<20)：震荡市不加减分，但在综合判断中降低信号可靠性

    # ========== V5.9: RSI背离检测 ==========
    # 8. RSI背离（最高优先级反转信号，权重 1.5 分）
    div_type = divergence_data.get("type", "none")
    div_desc = divergence_data.get("description", "")
    if div_type == "bullish_divergence":
        buy_reasons.append(f"RSI底背离：{div_desc}")
        buy_count += 1.5
    elif div_type == "bearish_divergence":
        sell_reasons.append(f"RSI顶背离：{div_desc}")
        sell_count += 1.5

    # ========== V5.15: K线形态识别 ==========
    # 9. K线反转形态（权重 1.0~2.0 分）
    candle_patterns = detect_candlestick_patterns(data)
    for cp in candle_patterns:
        if cp["type"] == "bullish":
            buy_reasons.append(f"K线{cp['pattern']}：{cp['desc']}")
            buy_count += cp["score"] * 0.5
        elif cp["type"] == "bearish":
            sell_reasons.append(f"K线{cp['pattern']}：{cp['desc']}")
            sell_count += abs(cp["score"]) * 0.5

    # ========== V5.15: 成交量背离检测 ==========
    # 10. 价量背离（权重 1.0~1.5 分）
    vol_divs = detect_volume_divergence(data['Close'], data['Volume'])
    for vd in vol_divs:
        if vd["type"] == "bullish":
            buy_reasons.append(f"成交量背离：{vd['desc']}")
            buy_count += vd["score"] * 0.5
        elif vd["type"] == "bearish":
            sell_reasons.append(f"成交量背离：{vd['desc']}")
            sell_count += abs(vd["score"]) * 0.5

    # ========== V5.18: 资金流向指标 ==========
    # 11. OBV背离（权重 1.5 分）
    if obv_data["divergence"] == "bullish":
        buy_reasons.append(f"OBV底背离：{obv_data['desc']}")
        buy_count += 1.5
    elif obv_data["divergence"] == "bearish":
        sell_reasons.append(f"OBV顶背离：{obv_data['desc']}")
        sell_count += 1.5

    # 12. MFI超买超卖（权重 1.0 分）
    if mfi_data["zone"] == "超卖":
        buy_reasons.append(f"MFI={mfi_data['mfi']}，资金流量极度超卖")
        buy_count += 1.0
    elif mfi_data["zone"] == "超买":
        sell_reasons.append(f"MFI={mfi_data['mfi']}，资金流量极度超买")
        sell_count += 1.0

    # 13. CMF资金流向（权重 1.0 分）
    if cmf_data["cmf"] > 0.15:
        buy_reasons.append(f"CMF={cmf_data['cmf']}，资金持续流入（买方主导）")
        buy_count += 1.0
    elif cmf_data["cmf"] < -0.15:
        sell_reasons.append(f"CMF={cmf_data['cmf']}，资金持续流出（卖方主导）")
        sell_count += 1.0

    # 14. VWAP位置（权重 0.5 分）
    if vwap_data["position"] == "above" and abs(vwap_data["distance"]) > 2:
        buy_reasons.append(f"价格高于VWAP（+{vwap_data['distance']}%），买方主导")
        buy_count += 0.5
    elif vwap_data["position"] == "below" and abs(vwap_data["distance"]) > 2:
        sell_reasons.append(f"价格低于VWAP（{vwap_data['distance']}%），卖方主导")
        sell_count += 0.5

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

    # 建议价格（基于ATR动态止损 + 近期高低点）
    recent_low = data['Low'].tail(20).min()
    recent_high = data['High'].tail(20).max()
    # V5.16: 改进ATR计算（14日真实波幅均值，替代单点范围/14）
    highs = data['High'].tail(15).values
    lows = data['Low'].tail(15).values
    recent_closes = data['Close'].tail(15).values
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - recent_closes[i-1]),
            abs(lows[i] - recent_closes[i-1])
        )
        trs.append(tr)
    atr = round(sum(trs) / len(trs), 2) if trs else 0

    if trade_point in ("strong_buy", "buy"):
        entry_price = round(current_price * 0.995, 2)       # 稍低于当前价
        stop_loss = round(entry_price - atr * 2, 2)         # 2倍ATR动态止损
        take_profit = round(entry_price + atr * 3, 2)       # 3倍ATR止盈
    elif trade_point in ("strong_sell", "sell"):
        # V5.17.4: sell 信号也给出完整价格建议（不再设0）
        entry_price = round(current_price * 0.995, 2)       # 建议卖出价（稍低于当前价作为保守估计）
        stop_loss = round(entry_price + atr * 2, 2)         # 反弹止损位（价格涨破此位则卖出信号失效）
        take_profit = round(recent_low * 1.02, 2)           # 目标止盈价（接近近期低点）
    else:
        entry_price = round(current_price * 0.995, 2)       # V5.16: 观望也给出入场参考
        stop_loss = round(entry_price - atr * 2, 2)         # 2倍ATR动态止损
        take_profit = round(entry_price + atr * 3, 2)       # 3倍ATR止盈

    return {
        "trade_point": trade_point,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "atr": atr,                                   # V5.16: 14日ATR波动率
        "score": score,
        "adx": round(adx_val, 2),                   # V5.9: ADX数值
        "adx_trend": adx_trend,                      # V5.9: ADX趋势类型
        "plus_di": round(plus_di, 2),                # V5.9: +DI
        "minus_di": round(minus_di, 2),              # V5.9: -DI
        "rsi_divergence_type": div_type,             # V5.9: RSI背离类型
        "rsi_divergence_desc": div_desc,             # V5.9: RSI背离描述
    }


# V5.14: 大盘指数映射（市场 → yfinance 代码 + 名称）
_MARKET_INDEX = {
    "us":  {"symbol": "^GSPC", "name": "标普500"},
    "hk":  {"symbol": "^HSI",  "name": "恒生指数"},
    "cn":  {"symbol": "000001.SS", "name": "上证指数"},
}


def get_market_trend(market):
    """
    V5.14: 获取大盘指数趋势，用于评估个股信号的可靠性。
    
    返回: {"trend": "bullish"|"bearish"|"neutral", 
           "change_percent": float, "ma_position": str, "label": str}
    失败时返回 None
    """
    if market not in _MARKET_INDEX:
        return None
    
    idx = _MARKET_INDEX[market]
    try:
        _rate_limit_wait()
        ticker = yf.Ticker(idx["symbol"])
        data = ticker.history(period="3mo")
        if data.empty or len(data) < 20:
            return None
        
        close = data['Close']
        current = close.iloc[-1]
        prev = close.iloc[-2] if len(data) > 1 else current
        change_pct = round((current - prev) / prev * 100, 2)
        ma20 = close.rolling(20).mean().iloc[-1]
        
        # 趋势判断：价格vsMA20 + 近5日方向
        recent_trend = sum(1 for i in range(-5, 0) if i >= -len(close) and close.iloc[i] > close.iloc[i-1])
        price_above_ma = current > ma20 if not pd.isna(ma20) else None
        
        if price_above_ma is True and recent_trend >= 3:
            trend = "bullish"
            label = "上涨趋势"
        elif price_above_ma is False and recent_trend <= 2:
            trend = "bearish"
            label = "下跌趋势"
        else:
            trend = "neutral"
            label = "震荡"

        vs_ma = "上方" if price_above_ma else "下方" if price_above_ma is not None else "≈"
        return {
            "trend": trend,
            "change_percent": change_pct,
            "ma_position": vs_ma,
            "label": label,
            "name": idx["name"],
        }
    except Exception:
        return None


def get_period_trend(data):
    """
    V5.14: 对单个周期的K线数据做简化趋势判断。
    用于周线/月线的多周期共振分析。

    返回: {"trend": "bullish"|"bearish"|"neutral", "rsi": float, "price_vs_ma20": str, "label": str}
    """
    if len(data) < 20:
        return {"trend": "neutral", "rsi": 50, "price_vs_ma20": "—", "label": "数据不足"}

    close = data['Close']
    current = close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1] if len(data) >= 20 else current

    # 简化 RSI
    rsi_val = 50
    try:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if loss != 0 and not pd.isna(loss):
            rs = gain / loss
            rsi_val = round(100 - (100 / (1 + rs)), 1)
    except Exception:
        pass

    price_above_ma = current > ma20 if not pd.isna(ma20) else None

    if rsi_val > 55 and price_above_ma is True:
        trend = "bullish"
        label = "看多"
    elif rsi_val < 45 and price_above_ma is False:
        trend = "bearish"
        label = "看空"
    else:
        trend = "neutral"
        label = "震荡"

    vs_ma = f"价格{'高于' if price_above_ma else '低于' if price_above_ma is not None else '≈'}MA20"
    return {"trend": trend, "rsi": rsi_val, "price_vs_ma20": vs_ma, "label": label}


def detect_candlestick_patterns(data):
    """
    V5.15: K线形态识别。
    识别锤子线、吞没形态、十字星、启明星/黄昏星等常见反转形态。
    
    返回：[{"pattern": str, "type": "bullish"|"bearish", "desc": str, "score": int}, ...]
    """
    patterns = []
    if len(data) < 4:
        return patterns
    
    o = data['Open']
    h = data['High']
    l = data['Low']
    c = data['Close']
    
    def body(i):
        return abs(c.iloc[i] - o.iloc[i])
    
    def upper_shadow(i):
        return h.iloc[i] - max(c.iloc[i], o.iloc[i])
    
    def lower_shadow(i):
        return min(c.iloc[i], o.iloc[i]) - l.iloc[i]
    
    def total_range(i):
        return h.iloc[i] - l.iloc[i]
    
    def is_bullish(i):
        return c.iloc[i] > o.iloc[i]
    
    def is_bearish(i):
        return c.iloc[i] < o.iloc[i]
    
    i = -1  # 只看最新一根 K 线
    
    # 跳过无效数据
    if total_range(i) <= 0 or body(i) <= 0:
        return patterns
    
    body_ratio = body(i) / total_range(i)
    us_ratio = upper_shadow(i) / total_range(i)
    ls_ratio = lower_shadow(i) / total_range(i)
    
    # 1. 锤子线（Hammer）：小实体在顶部，长下影，几乎无上影
    if ls_ratio > 0.6 and us_ratio < 0.15 and body_ratio < 0.3:
        patterns.append({
            "pattern": "锤子线",
            "type": "bullish",
            "desc": "长下影线表明下方支撑强劲，可能止跌反弹",
            "score": 2
        })
    
    # 2. 倒锤子线（Inverted Hammer）/ 流星线（Shooting Star）
    if us_ratio > 0.6 and ls_ratio < 0.15 and body_ratio < 0.3:
        if is_bullish(i):
            patterns.append({
                "pattern": "倒锤子线",
                "type": "bullish",
                "desc": "连续下跌后出现，可能变盘向上",
                "score": 2
            })
        else:
            patterns.append({
                "pattern": "流星线",
                "type": "bearish",
                "desc": "连续上涨后出现，长上影表明抛压出现，可能见顶",
                "score": -2
            })
    
    # 3. 十字星（Doji）：开收价接近
    if body_ratio < 0.05:
        patterns.append({
            "pattern": "十字星",
            "type": "neutral",
            "desc": "多空力量均衡，往往预示趋势转折",
            "score": 0  # 方向由上下文决定，暂不直接加分
        })
    
    # 4. 吞没形态（需要前一根 K 线）
    if len(data) >= 2:
        prev_body = body(-2)
        prev_open = o.iloc[-2]
        prev_close = c.iloc[-2]
        
        # 看涨吞没：前阴后阳，阳线实体完全包住阴线
        if (is_bearish(-2) and is_bullish(-1) and
            o.iloc[-1] <= prev_close and c.iloc[-1] >= prev_open and
            body(-1) > prev_body * 1.2):
            patterns.append({
                "pattern": "看涨吞没",
                "type": "bullish",
                "desc": "阳线完全吞没前日阴线，多头强势反攻",
                "score": 3
            })
        
        # 看跌吞没：前阳后阴，阴线实体完全包住阳线
        if (is_bullish(-2) and is_bearish(-1) and
            o.iloc[-1] >= prev_close and c.iloc[-1] <= prev_open and
            body(-1) > prev_body * 1.2):
            patterns.append({
                "pattern": "看跌吞没",
                "type": "bearish",
                "desc": "阴线完全吞没前日阳线，空头强势反攻",
                "score": -3
            })
    
    # 5. 启明星/黄昏星（需要前两根 K 线）
    if len(data) >= 3:
        p2_body = body(-3)
        p1_body = body(-2)
        
        # 启明星：大阴线 + 小实体（跳空低开） + 大阳线
        if (is_bearish(-3) and p2_body > total_range(-3) * 0.4 and
            body(-2) < total_range(-2) * 0.25 and
            is_bullish(-1) and body(-1) > total_range(-1) * 0.4 and
            c.iloc[-1] > (o.iloc[-3] + c.iloc[-3]) / 2):
            patterns.append({
                "pattern": "启明星",
                "type": "bullish",
                "desc": "三线反转形态，大阴→小星→大阳，强烈看涨",
                "score": 4
            })
        
        # 黄昏星：大阳线 + 小实体（跳空高开） + 大阴线
        if (is_bullish(-3) and p2_body > total_range(-3) * 0.4 and
            body(-2) < total_range(-2) * 0.25 and
            is_bearish(-1) and body(-1) > total_range(-1) * 0.4 and
            c.iloc[-1] < (o.iloc[-3] + c.iloc[-3]) / 2):
            patterns.append({
                "pattern": "黄昏星",
                "type": "bearish",
                "desc": "三线反转形态，大阳→小星→大阴，强烈看跌",
                "score": -4
            })
    
    return patterns


def detect_volume_divergence(closes, volumes):
    """
    V5.15: 成交量背离检测。
    
    价涨量缩 → 上涨乏力（bearish divergence）
    价跌量缩 → 抛压衰竭（bullish divergence）
    
    返回：[{"type": "bullish"|"bearish", "desc": str, "score": int}, ...]
    """
    divergences = []
    if len(closes) < 10 or len(volumes) < 10:
        return divergences
    
    # 最近 5 个交易日
    c5 = closes.iloc[-5:]
    v5 = volumes.iloc[-5:]
    
    price_change = (c5.iloc[-1] - c5.iloc[0]) / c5.iloc[0] * 100
    vol_change = (v5.iloc[-1] - v5.iloc[0]) / v5.iloc[0] * 100 if v5.iloc[0] > 0 else 0
    
    # 近5日均量 vs 前5日均量
    v_recent = volumes.iloc[-5:].mean()
    v_prev = volumes.iloc[-10:-5].mean() if len(volumes) >= 10 else v_recent
    vol_trend = (v_recent - v_prev) / v_prev * 100 if v_prev > 0 else 0
    
    # 1. 价涨量缩：上涨乏力（bearish）
    if price_change > 2 and vol_trend < -10:
        divergences.append({
            "type": "bearish",
            "desc": f"近5日价格涨{price_change:+.1f}%，但成交量萎缩{vol_trend:.0f}%，上涨动能不足",
            "score": -3
        })
    elif price_change > 1 and vol_trend < -15:
        divergences.append({
            "type": "bearish",
            "desc": f"价涨量缩（价格{price_change:+.1f}%，量{vol_trend:.0f}%），警惕冲高回落",
            "score": -2
        })
    
    # 2. 价跌量缩：抛压衰竭（bullish）
    if price_change < -2 and vol_trend < -10:
        divergences.append({
            "type": "bullish",
            "desc": f"近5日价格跌{price_change:.1f}%，但成交量同步萎缩{vol_trend:.0f}%，抛压在减少",
            "score": 3
        })
    elif price_change < -1 and vol_trend < -15:
        divergences.append({
            "type": "bullish",
            "desc": f"价跌量缩（价格{price_change:.1f}%，量{vol_trend:.0f}%），抛压衰减中",
            "score": 2
        })
    
    return divergences


# ==================== V5.18: 资金流向技术指标 ====================

def calculate_obv(data):
    """
    On-Balance Volume (OBV) — 能量潮

    涨日：OBV += 成交量；跌日：OBV -= 成交量；平盘：不变
    返回：OBV背离状态（价格与OBV走势背离=聪明钱反向操作）
    """
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    obv_series = pd.Series(obv, index=data.index)

    # 近20日价格和OBV趋势
    price_20d = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100 if len(data) >= 20 else 0
    obv_base = max(abs(obv_series.iloc[-20]), 1) if len(data) >= 20 else 1
    obv_20d = ((obv_series.iloc[-1] - obv_series.iloc[-20]) / obv_base * 100) if len(data) >= 20 else 0

    divergence = "none"
    div_desc = ""
    if price_20d > 3 and obv_20d < -5:
        divergence = "bearish"
        div_desc = f"OBV顶背离（价涨{price_20d}%但OBV跌{abs(obv_20d)}%，资金暗中撤退）"
    elif price_20d < -3 and obv_20d > 5:
        divergence = "bullish"
        div_desc = f"OBV底背离（价跌{abs(price_20d)}%但OBV涨{obv_20d}%，资金暗中吸筹）"
    elif len(data) >= 20 and obv_20d > 10:
        div_desc = f"OBV持续流入（近20日+{obv_20d}%），资金积极进场"

    return {
        "divergence": divergence,
        "desc": div_desc,
        "obv_20d": round(obv_20d, 1),
    }


def calculate_mfi(data, period=14):
    """
    Money Flow Index (MFI) — 资金流量指数

    价格+成交量的RSI升级版。>80=超买资金出逃，<20=超卖资金回流
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']

    positive_flow = []
    negative_flow = []
    for i in range(1, len(data)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    pf_series = pd.Series(positive_flow, index=data.index[1:])
    nf_series = pd.Series(negative_flow, index=data.index[1:])

    if len(pf_series) < period:
        return {"mfi": 50, "zone": "中性"}

    pf_sum = pf_series.rolling(period).sum()
    nf_sum = nf_series.rolling(period).sum()
    mfi_ratio = pf_sum / nf_sum.replace(0, float('nan'))
    mfi_series = 100 - (100 / (1 + mfi_ratio))

    mfi_val = round(mfi_series.iloc[-1], 1) if not pd.isna(mfi_series.iloc[-1]) else 50
    zone = "超买" if mfi_val > 80 else ("超卖" if mfi_val < 20 else "中性")
    return {"mfi": mfi_val, "zone": zone}


def calculate_cmf(data, period=20):
    """
    Chaikin Money Flow (CMF) — 蔡金资金流

    用收盘价在当日高低区的位置判断资金流入强度
    CMF > 0.1 = 持续流入，CMF < -0.1 = 持续流出
    """
    hl_range = data['High'] - data['Low']
    mf_mult = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / hl_range.replace(0, float('nan'))
    mf_volume = mf_mult * data['Volume']
    cmf_series = mf_volume.rolling(period).sum() / data['Volume'].rolling(period).sum()
    cmf_val = round(cmf_series.iloc[-1], 3) if not pd.isna(cmf_series.iloc[-1]) else 0
    return {"cmf": cmf_val}


def calculate_vwap(data):
    """
    Volume Weighted Average Price (VWAP) — 成交量加权均价

    机构常用基准价。价格 > VWAP = 买方主导；价格 < VWAP = 卖方主导
    """
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    cum_pv = (typical_price * data['Volume']).sum()
    cum_vol = data['Volume'].sum()
    vwap = cum_pv / cum_vol if cum_vol > 0 else data['Close'].iloc[-1]
    current = data['Close'].iloc[-1]
    distance = round((current - vwap) / vwap * 100, 2)
    position = "above" if current > vwap else "below"
    return {"vwap": round(vwap, 2), "position": position, "distance": distance}


def compute_signal_accuracy(data):
    """
    V5.17: 信号准确率回测 — 使用与 detect_trade_points 完全一致的10维评分逻辑。
    
    在最近60个交易日窗口内，逐日使用完整的10维打分体系判定买卖信号，
    追踪信号触发后5日的实际走势，统计方向一致率。
    
    10个维度：MACD金叉/死叉、RSI超卖/超买、KDJ极端值、成交量、布林带、
    均线回踩、ADX趋势强度、RSI背离、K线形态、成交量背离。
    
    只统计有明确方向信号的日子（trade_point != hold），中性日跳过。
    """
    closes = data['Close']
    highs = data['High']
    lows = data['Low']
    opens = data['Open']
    volumes = data['Volume']
    
    n = len(closes)
    if n < 30:
        return {"accuracy": None, "testable_days": 0, "consistent_days": 0, "note": "数据不足30日，无法回测"}
    
    lookback = min(60, n - 6)
    if lookback < 5:
        return {"accuracy": None, "testable_days": 0, "consistent_days": 0, "note": "数据量不足，无法回测"}
    
    # ---- 预计算全序列指标 ----
    # RSI
    delta = closes.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs_series = gain / loss.replace(0, float('nan'))
    rsi_series = 100 - (100 / (1 + rs_series))
    
    # MACD
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_series = ema12 - ema26
    sig_series = macd_series.ewm(span=9, adjust=False).mean()
    hist_series = (macd_series - sig_series) * 2
    
    # KDJ
    low9 = lows.rolling(9).min()
    high9 = highs.rolling(9).max()
    rsv = ((closes - low9) / (high9 - low9).replace(0, float('nan'))) * 100
    k_series = rsv.ewm(com=2, adjust=False).mean()
    d_series = k_series.ewm(com=2, adjust=False).mean()
    j_series = 3 * k_series - 2 * d_series
    
    # Bollinger
    ma20_series = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    upper_series = ma20_series + 2 * std20
    lower_series = ma20_series - 2 * std20
    
    # MA
    ma5_series = closes.rolling(5).mean()
    ma10_series = closes.rolling(10).mean()
    ma50_series = closes.rolling(50).mean()
    
    # Volume ratio
    vol_ma10 = volumes.rolling(10).mean()
    
    consistent = 0
    total = 0
    
    # 回测起始位置（至少需要50根K线做MA50+RSI背离+ADX计算）
    start = max(n - lookback - 5, 50)
    if start >= n - 5:
        return {"accuracy": None, "testable_days": 0, "consistent_days": 0, "note": "数据量不足（需>=50日），无法回测"}
    
    for i in range(start, n - 5):
        buy_count = 0.0
        sell_count = 0.0
        
        # 当日数据切片（用于调用与detect_trade_points相同的函数）
        sub_data = data.iloc[:i + 1]
        
        rsi_i = rsi_series.iloc[i]
        rsi_prev = rsi_series.iloc[i - 1] if i > 0 else rsi_i
        curr_close = closes.iloc[i]
        curr_open = opens.iloc[i]
        curr_high = highs.iloc[i]
        curr_low = lows.iloc[i]
        is_green = curr_close > curr_open
        prev_close = closes.iloc[i - 1] if i > 0 else curr_close
        
        # ========== 1. MACD金叉/死叉（对齐detect_trade_points） ==========
        macd_i = macd_series.iloc[i]
        sig_i = sig_series.iloc[i]
        hist_i = hist_series.iloc[i]
        if i > 0 and not pd.isna(macd_i):
            prev_macd = macd_series.iloc[i - 1]
            prev_sig = sig_series.iloc[i - 1]
            # 金叉：DIF从下穿越DEA
            if prev_macd <= prev_sig and macd_i > sig_i:
                buy_count += 1.0
            # 死叉：DIF从上穿越DEA
            if prev_macd >= prev_sig and macd_i < sig_i:
                sell_count += 1.0
        
        # ========== 2. RSI超卖/超买（对齐detect_trade_points） ==========
        if not pd.isna(rsi_i):
            if rsi_prev < 30 and rsi_i >= 30:
                buy_count += 1.0
            elif rsi_i < 20:
                buy_count += 1.0
            if rsi_prev > 70 and rsi_i <= 70:
                sell_count += 1.0
            elif rsi_i > 85:
                sell_count += 1.0
        
        # ========== 3. KDJ极端值+金叉/死叉（对齐detect_trade_points） ==========
        j_i = j_series.iloc[i]
        k_i = k_series.iloc[i]
        d_i = d_series.iloc[i]
        if not pd.isna(j_i) and i > 0:
            prev_k = k_series.iloc[i - 1]
            prev_d = d_series.iloc[i - 1]
            kdj_golden = (prev_k < prev_d and k_i >= d_i)
            kdj_death = (prev_k > prev_d and k_i <= d_i)
            if j_i < 0 and kdj_golden:
                buy_count += 1.0
            elif j_i < 0:
                buy_count += 0.5
            if j_i > 100 and kdj_death:
                sell_count += 1.0
            elif j_i > 100:
                sell_count += 0.5
        
        # ========== 4. 放量上涨/下跌（对齐detect_trade_points） ==========
        vol_ratio_i = volumes.iloc[i] / vol_ma10.iloc[i] if not pd.isna(vol_ma10.iloc[i]) and vol_ma10.iloc[i] > 0 else 1
        if vol_ratio_i > 1.2:
            if is_green:
                buy_count += 1.0
            else:
                sell_count += 1.0
        
        # ========== 5. 布林带反弹/回落（对齐detect_trade_points） ==========
        upper_i = upper_series.iloc[i]
        lower_i = lower_series.iloc[i]
        if not pd.isna(lower_i) and i > 0:
            prev_low = lows.iloc[i - 1]
            prev_high = highs.iloc[i - 1]
            if prev_low <= lower_i and is_green:
                buy_count += 1.0
            elif curr_close <= lower_i * 1.01 and is_green:
                buy_count += 0.5
            if prev_high >= upper_i and not is_green:
                sell_count += 1.0
            elif curr_close >= upper_i * 0.99 and not is_green:
                sell_count += 0.5
        
        # ========== 6. 均线回踩（对齐detect_trade_points） ==========
        ma5_i = ma5_series.iloc[i]
        ma10_i = ma10_series.iloc[i]
        ma20_i = ma20_series.iloc[i]
        ma50_i = ma50_series.iloc[i]
        if not pd.isna(ma5_i) and not pd.isna(ma10_i) and not pd.isna(ma20_i):
            ma_bullish = curr_close > ma5_i and ma5_i > ma10_i and ma10_i > ma20_i
            if ma_bullish and prev_close <= ma20_i and curr_close > ma20_i:
                buy_count += 1.0
            ma_bearish = curr_close < ma5_i and ma5_i < ma10_i and ma10_i < ma20_i
            if ma_bearish and prev_close >= ma20_i and curr_close < ma20_i:
                sell_count += 1.0
        
        # ========== 7. ADX趋势强度（对齐detect_trade_points） ==========
        try:
            adx_d = calculate_adx(sub_data)
            adx_trend = adx_d.get("trend", "ranging")
            adx_val = adx_d.get("adx", 0)
            adx_plus = adx_d.get("plus_di", 0)
            adx_minus = adx_d.get("minus_di", 0)
            if adx_trend == "strong_bull":
                buy_count += 1.0
            elif adx_trend == "strong_bear":
                sell_count += 1.0
            elif adx_trend == "weak_bull":
                buy_count += 0.5
            elif adx_trend == "weak_bear":
                sell_count += 0.5
        except Exception:
            pass
        
        # ========== 8. RSI背离（对齐detect_trade_points） ==========
        try:
            div_d = detect_rsi_divergence(sub_data)
            div_type = div_d.get("type", "none")
            if div_type == "bullish_divergence":
                buy_count += 1.5
            elif div_type == "bearish_divergence":
                sell_count += 1.5
        except Exception:
            pass
        
        # ========== 9. K线形态（对齐detect_trade_points） ==========
        try:
            cps = detect_candlestick_patterns(sub_data)
            for cp in cps:
                if cp["type"] == "bullish":
                    buy_count += cp["score"] * 0.5
                else:
                    sell_count += abs(cp["score"]) * 0.5
        except Exception:
            pass
        
        # ========== 10. 成交量背离（对齐detect_trade_points） ==========
        try:
            # 使用近30日数据做量价背离检测
            win_start = max(0, i - 29)
            sub_closes = closes.iloc[win_start:i + 1]
            sub_volumes = volumes.iloc[win_start:i + 1]
            vds = detect_volume_divergence(sub_closes, sub_volumes)
            for vd in vds:
                if vd["type"] == "bullish":
                    buy_count += vd["score"] * 0.5
                else:
                    sell_count += abs(vd["score"]) * 0.5
        except Exception:
            pass
        
        # ========== 信号判定（与detect_trade_points完全一致） ==========
        if buy_count >= 2 and sell_count == 0:
            direction = "bull"
        elif buy_count >= 1.5 and buy_count > sell_count:
            direction = "bull"
        elif sell_count >= 2 and buy_count == 0:
            direction = "bear"
        elif sell_count >= 1.5 and sell_count > buy_count:
            direction = "bear"
        else:
            continue  # 中性日跳过
        
        # 5日后实际涨跌
        forward_return = closes.iloc[i + 5] - closes.iloc[i]
        total += 1
        if (direction == "bull" and forward_return > 0) or (direction == "bear" and forward_return < 0):
            consistent += 1
    
    if total == 0:
        return {"accuracy": None, "testable_days": 0, "consistent_days": 0, 
                "note": f"回测窗口{min(60, n-6)}日中无明确信号触发"}
    
    accuracy = round(consistent / total * 100, 1)
    return {
        "accuracy": accuracy,
        "testable_days": total,
        "consistent_days": consistent,
        "note": f"基于10维完整信号回测（对齐detect_trade_points），有效信号日{total}天，5日方向一致率{accuracy}%"
    }


def build_formatted_report(
    name, symbol, market, currency,
    current_price, change_percent,
    signal, confidence, trade_point, trade_point_cn,
    trade_score, buy_reasons_text, sell_reasons_text,
    entry_price, stop_loss, take_profit,
    rsi, rsi_prev, rsi_delta,
    macd_val, macd_signal, macd_hist, macd_cross,
    kdj_k, kdj_d, kdj_j,
    adx, adx_trend, plus_di, minus_di,
    boll_upper, boll_middle, boll_lower,
    ma5, ma10, ma20, ma50,
    volume_signal, volume_ratio, trend_direction,
    support_level, resistance_level,
    kline_text, rsi_div_type, rsi_div_desc,
    accuracy_data=None, weekly_trend=None, monthly_trend=None, market_trend=None,
    candle_patterns=None, vol_divergences=None, atr=None, fundamentals=None,
    obv_data=None, mfi=None, cmf=None, vwap_data=None
):
    """
    V5.14: API层预渲染完整中文分析报告（含多周期共振+大盘环境+信号准确率回测）。
    由 API 直接生成格式化报告文本，Agent 只做管道转发，彻底消灭 Agent 创作空间。
    """
    lines = []
    lines.append(f"📊 {name}（{symbol}）技术分析报告")
    lines.append("")
    # 价格行
    change_sign = "+" if change_percent >= 0 else ""
    currency_symbol = "HK$" if currency == "HKD" else ("$" if currency == "USD" else "¥")
    lines.append(f"💰 当前价格：{currency_symbol}{current_price}（{change_sign}{change_percent}%）")
    # 信号 + 星级
    signal_cn_map = {
        "strong_buy": "强烈买入",
        "buy": "看多",
        "hold": "观望",
        "sell": "看空",
        "strong_sell": "强烈看空",
    }
    signal_cn = signal_cn_map.get(signal.lower(), "观望")
    # 星级（V5.18.2: 基于 trade_point 而非 trade_score，与底部评分各司其职）
    stars_map = {
        "strong_buy": "★★★",
        "strong_sell": "★★★",
        "buy": "★★☆",
        "sell": "★★☆",
        "hold": "— — —"
    }
    stars = stars_map.get(trade_point.lower(), "— — —")
    conf_cn = {"high": "强", "medium": "中等", "low": "弱"}.get(confidence, confidence)  # V5.18.2: 兼容调用方传中文标签
    # V5.17.6: ADX<25 震荡市过滤 — 趋势不明朗时压制信号
    adx_filtered = adx < 25 and signal_cn != "观望"
    if adx_filtered:
        signal_cn = "震荡观望"
        stars = "— — —"
        conf_cn = "弱"
    lines.append(f"📌 综合信号：{signal_cn} {stars}（{conf_cn}）")
    if adx_filtered:
        lines.append(f"⚠️ ADX={adx}<25，趋势不明确，处于震荡市。以上评分仅供参考，不建议基于技术信号操作。")
    # 买卖点
    entry_str = f"{currency_symbol}{entry_price}" if entry_price and entry_price != 0 else "—"
    stop_str = f"{currency_symbol}{stop_loss}" if stop_loss and stop_loss != 0 else "—"
    take_str = f"{currency_symbol}{take_profit}" if take_profit and take_profit != 0 else "—"
    lines.append(f"🎯 买卖点：入场 {entry_str} / 止损 {stop_str} / 止盈 {take_str}")
    # V5.17.9: ADX<25 震荡市买卖点仅供参考
    if adx_filtered:
        lines.append(f"⚠️ ADX={adx}<25 趋势不明，以上价位仅供参考，不建议作为实际操作依据")
    # V5.16: 风险收益比
    if entry_price and stop_loss and take_profit and entry_price != 0 and stop_loss != 0:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk > 0:
            rr = round(reward / risk, 1)
            lines.append(f"📐 风险收益比：R:R = 1:{rr}（每承担{currency_symbol}1风险，预期收益{currency_symbol}{rr}）")
    lines.append("=" * 40)
    # RSI背离警告（P0）
    if rsi_div_type and rsi_div_type != "none":
        lines.append(f"⚠️ RSI背离信号：{rsi_div_desc}")
    # 极端信号
    if rsi > 85:
        lines.append(f"⚠️ 检测到极端超买指标（RSI {rsi}），建议等待信号冷却后再操作。")
    elif rsi < 15:
        lines.append(f"⚠️ 检测到极端超卖指标（RSI {rsi}），建议等待信号冷却后再操作。")
    lines.append("")
    # V5.14: 大盘环境因子
    market_coef = 1.0  # 默认无影响
    if market_trend:
        mt_name = market_trend.get("name", "大盘")
        mt_label = market_trend.get("label", "—")
        mt_change = market_trend.get("change_percent", 0)
        mt_pos = market_trend.get("ma_position", "—")
        lines.append("=" * 40)
        lines.append(f"🌍 大盘环境：{mt_name}")
        lines.append("=" * 40)

        if market_trend.get("trend") == "bullish":
            lines.append(f"  走势：{mt_label}（{mt_change:+.2f}%，价格在MA20{mt_pos}）")
            lines.append(f"  → 大盘强势，个股买入信号可信度+10%，卖出信号需谨慎")
            market_coef = 1.1
        elif market_trend.get("trend") == "bearish":
            lines.append(f"  走势：{mt_label}（{mt_change:+.2f}%，价格在MA20{mt_pos}）")
            lines.append(f"  → 大盘弱势，个股买入信号可信度-15%，卖出信号可信度+10%")
            market_coef = 0.85
        else:
            lines.append(f"  走势：{mt_label}（{mt_change:+.2f}%，价格在MA20{mt_pos}）")
            lines.append(f"  → 大盘震荡，个股信号不受大盘影响")
        lines.append("")
    lines.append("=" * 40)
    lines.append("📊 技术评分明细（满分 ±100）")
    lines.append("=" * 40)
    # RSI背离评分
    div_score = 0
    if rsi_div_type == "bullish_divergence":
        div_score = 20
    elif rsi_div_type == "bearish_divergence":
        div_score = -20
    lines.append(f"  RSI背离：{div_score}分（{rsi_div_desc if rsi_div_type != 'none' else '无'}）")
    # V5.15: K线形态评分
    cp_score = 0
    cp_label = "无特殊形态"
    if candle_patterns:
        cp_parts = []
        for cp in candle_patterns:
            s = cp["score"] if cp["type"] != "neutral" else 0
            cp_score += s
            emoji = "🟢" if cp["type"] == "bullish" else ("🔴" if cp["type"] == "bearish" else "⚪")
            cp_parts.append(f"{emoji}{cp['pattern']}")
        cp_label = ",".join(cp_parts) if cp_parts else "无特殊形态"
    lines.append(f"  K线形态：{cp_score}分（{cp_label}）")
    # V5.15: 成交量背离评分
    vd_score = 0
    vd_label = "无背离"
    if vol_divergences:
        vd_parts = []
        for vd in vol_divergences:
            vd_score += vd["score"]
            emoji = "🟢" if vd["type"] == "bullish" else "🔴"
            vd_parts.append(f"{emoji}{vd['type']}")
        vd_label = ",".join(vd_parts) if vd_parts else "无背离"
    lines.append(f"  成交量背离：{vd_score}分（{vd_label}）")
    # ADX
    adx_score = 0
    if adx_trend == "strong_bull":
        adx_score = 15
    elif adx_trend == "weak_bull":
        adx_score = 8
    elif adx_trend == "strong_bear":
        adx_score = -15
    elif adx_trend == "weak_bear":
        adx_score = -8
    adx_label = {"strong_bull": "强势多头", "weak_bull": "偏多", "strong_bear": "强势空头", "weak_bear": "偏空", "ranging": "震荡"}.get(adx_trend, "震荡")
    adx_note = "⚠️ADX<25=震荡市，综合信号已过滤" if adx_filtered else ""
    lines.append(f"  ADX趋势：{adx_score}分（ADX={adx}，{adx_label}，+DI={plus_di}/-DI={minus_di}）{adx_note}")
    # MACD
    macd_score = 0
    if macd_cross == "golden":
        macd_score = 15
    elif macd_val > macd_signal and macd_hist > 0:
        macd_score = 10
    elif macd_cross == "death":
        macd_score = -15
    elif macd_val < macd_signal and macd_hist < 0:
        macd_score = -10
    macd_label = {"golden": "金叉", "death": "死叉", "none": "多头运行" if macd_val > macd_signal else "空头运行"}.get(macd_cross, "")
    lines.append(f"  MACD：{macd_score}分（{macd_label}，柱={macd_hist}）")
    # KDJ
    kdj_score = 0
    if kdj_k > kdj_d:
        kdj_score = 5
    if kdj_k < 20 and kdj_d < 20:
        kdj_score += 10
    elif kdj_k > 80 and kdj_d > 80:
        kdj_score -= 10
    elif kdj_k < kdj_d:
        kdj_score -= 5  # 死叉减分（与下方J值极端判定叠加）
    # J值极端超卖/超买（与 detect_trade_points 对齐，避免评分矛盾）
    if kdj_j < 0:
        kdj_score += 4  # J负值=深度超卖，反弹在即
    elif kdj_j > 100:
        kdj_score -= 4  # J>100=极度超买，回调风险
    kdj_zone = "超卖区" if kdj_k < 20 else ("超买区" if kdj_k > 80 else "中性")
    kdj_cross = "金叉" if kdj_k > kdj_d else "死叉"
    lines.append(f"  KDJ：{kdj_score}分（K={kdj_k}，D={kdj_d}（{kdj_zone}），K与D：{kdj_cross}）")
    # RSI（V5.14: 更细粒度，极端值获更高权重）
    rsi_score = 0
    if rsi < 10:
        rsi_score = 15
    elif rsi < 15:
        rsi_score = 12
    elif rsi < 20:
        rsi_score = 10
    elif rsi < 30:
        rsi_score = 7
    elif rsi > 90:
        rsi_score = -15
    elif rsi > 85:
        rsi_score = -12
    elif rsi > 80:
        rsi_score = -10
    elif rsi > 70:
        rsi_score = -7
    rsi_zone = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "正常")
    rsi_arrow = "▲" if rsi > rsi_prev else ("▼" if rsi < rsi_prev else "▬")
    lines.append(f"  RSI：{rsi_score}分（RSI={rsi}（{rsi_zone}）{rsi_arrow}较前日）")
    # 均线（V5.14: 弱化部分多头/空头的模糊信号）
    ma_score = 0
    if ma5 and ma10 and ma20:
        if ma5 > ma10 > ma20:
            ma_score = 15
        elif ma5 > ma10 or ma10 > ma20:
            ma_score = 5
        elif ma5 < ma10 < ma20:
            ma_score = -15
        elif ma5 < ma10 or ma10 < ma20:
            ma_score = -5
    ma_label = "多头排列" if ma5 and ma10 and ma20 and ma5 > ma10 > ma20 else ("空头排列" if ma5 and ma10 and ma20 and ma5 < ma10 < ma20 else "混杂")
    lines.append(f"  均线趋势：{ma_score}分（{ma_label}）")
    # 成交量
    vol_score = 0
    if volume_signal == "high_volume":
        vol_score = 10 if trend_direction == "bullish" else -10
    elif volume_signal == "above_avg":
        vol_score = 5 if trend_direction == "bullish" else -5
    elif volume_signal == "low_volume":
        vol_score = -3
    lines.append(f"  成交量：{vol_score}分（{volume_signal}，比率{volume_ratio}x）")
    # 布林带
    boll_score = 0
    if current_price <= boll_lower * 1.01:
        boll_score = 3
    elif current_price >= boll_upper * 0.99:
        boll_score = -3
    lines.append(f"  布林带：{boll_score}分（上轨{boll_upper}，下轨{boll_lower}）")
    # V5.18: 资金流向评分
    obv_score = 0
    obv_label = "无背离"
    if obv_data:
        if obv_data["divergence"] == "bullish":
            obv_score = 10
            obv_label = "🟢底背离（资金吸筹）"
        elif obv_data["divergence"] == "bearish":
            obv_score = -10
            obv_label = "🔴顶背离（资金撤退）"
        elif obv_data.get("desc"):
            obv_score = 3
            obv_label = "🟢持续流入"
    lines.append(f"  OBV能量潮：{obv_score}分（{obv_label}）")

    mfi_score = 0
    mfi_label = "中性"
    if mfi is not None:
        if mfi > 80:
            mfi_score = -10
            mfi_label = f"超买（{mfi}）"
        elif mfi < 20:
            mfi_score = 10
            mfi_label = f"超卖（{mfi}）"
        else:
            mfi_label = f"{mfi}"
    lines.append(f"  MFI资金流：{mfi_score}分（{mfi_label}）")

    cmf_score = 0
    cmf_label = "中性"
    if cmf is not None:
        if cmf > 0.15:
            cmf_score = 8
            cmf_label = f"🟢持续流入（{cmf}）"
        elif cmf < -0.15:
            cmf_score = -8
            cmf_label = f"🔴持续流出（{cmf}）"
        elif cmf > 0.05:
            cmf_score = 3
            cmf_label = f"🟢偏流入（{cmf}）"
        elif cmf < -0.05:
            cmf_score = -3
            cmf_label = f"🔴偏流出（{cmf}）"
        else:
            cmf_label = f"{cmf}"
    lines.append(f"  CMF蔡金流：{cmf_score}分（{cmf_label}）")

    vwap_score = 0
    vwap_label = f"${vwap_data['vwap']}" if vwap_data else "—"
    if vwap_data:
        if vwap_data["position"] == "above" and vwap_data["distance"] > 1:
            vwap_score = 3
            vwap_label = f"🟢高于VWAP（+{vwap_data['distance']}%）"
        elif vwap_data["position"] == "below" and abs(vwap_data["distance"]) > 1:
            vwap_score = -3
            vwap_label = f"🔴低于VWAP（{vwap_data['distance']}%）"
        else:
            vwap_label = f"${vwap_data['vwap']}（接近）"
    lines.append(f"  VWAP均价：{vwap_score}分（{vwap_label}）")
    lines.append("  " + "-" * 35)
    # V5.17.1: 多周期一致性折扣（提前计算，应用于总分）
    mt_discount = 0
    if weekly_trend and monthly_trend:
        week_label = weekly_trend.get("label", "—")
        month_label = monthly_trend.get("label", "—")
        periods = [("日线", signal_cn), ("周线", week_label), ("月线", month_label)]
        bullish_count = sum(1 for _, t in periods if "买" in t or "多" in t or "涨" in t or "强" in t)
        bearish_count = sum(1 for _, t in periods if "卖" in t or "空" in t or "跌" in t)
        if bullish_count == 3 or bearish_count == 3:
            mt_discount = 10  # 三线共振 +10
        elif bullish_count == 0 and bearish_count == 0:
            mt_discount = -5  # 三线分歧 -5
    total_score = div_score + adx_score + macd_score + kdj_score + rsi_score + ma_score + vol_score + boll_score + cp_score + vd_score + obv_score + mfi_score + cmf_score + vwap_score
    # V5.14: 大盘环境系数调整
    total_score = total_score * market_coef
    # V5.17.1: 多周期一致性折扣
    total_score = total_score + mt_discount
    total_score = max(-100, min(100, round(total_score, 0)))
    # V5.17.5: 评分行信号从 total_score 推导，与 10 维评分体系自洽
    if total_score >= 10:
        score_signal = "看多"
        score_stars = "★★★" if total_score >= 30 else "★★☆"
    elif total_score >= 3:
        score_signal = "偏多"
        score_stars = "★☆☆"
    elif total_score <= -10:
        score_signal = "看空"
        score_stars = "★★★" if total_score <= -30 else "★★☆"
    elif total_score <= -3:
        score_signal = "偏空"
        score_stars = "★☆☆"
    else:
        score_signal = "中性"
        score_stars = "— — —"
    mt_notes = []
    if market_coef != 1.0:
        mt_notes.append(f"大盘{market_coef:.0%}系数")
    if mt_discount != 0:
        mt_notes.append(f"多周期{"+" if mt_discount > 0 else ""}{mt_discount}分")
    mt_note = f"（含{','.join(mt_notes)}调整）" if mt_notes else ""
    # V5.17.8: ADX<25 时评分行标注"被ADX震荡市过滤"，避免与综合信号矛盾
    score_line = f"  总分：{int(total_score)} → {score_signal} {score_stars}{mt_note}"
    if adx_filtered:
        score_line += f"⚠️（ADX={adx}<25，综合信号已过滤为震荡观望）"
    lines.append(score_line)
    lines.append("=" * 40)
    lines.append("")
    lines.append("关键信号触发原因")
    # V5.17.9: ADX<25 时加一句上下文，避免利空因素一边倒看起来"很悲观"
    if adx_filtered and (buy_reasons_text or sell_reasons_text):
        lines.append(f"  💡 ADX={adx}<25 震荡市，以下技术信号可信度降低，仅供参考：")
    # V5.17.9: ADX<25 震荡市过滤时，跟踪存活买入理由 + 展示平衡
    if buy_reasons_text:
        visible_buy = []
        for r in buy_reasons_text.split("；"):
            r = r.strip()
            if not r:
                continue
            if adx_filtered and "ADX" in r:
                continue  # ADX<25 趋势不明，不展示"ADX偏多"等
            visible_buy.append(r)
            star = "★★★" if ("强烈" in r or "金叉" in r) else "★★☆"
            lines.append(f"  ✅ {r} {star}")
        # ADX 过滤后所有买入理由被压制 → 给上下文解释，避免沉默误导
        if not visible_buy and adx_filtered:
            lines.append(f"  💡 原始技术信号偏多，但 ADX={adx}<25 趋势不明确，综合降级为震荡观望")
    elif adx_filtered:
        lines.append(f"  ✅ 无明显买入信号（ADX={adx}<25，震荡市谨慎操作）")
    if sell_reasons_text:
        for r in sell_reasons_text.split("；"):
            if r.strip():
                star = "★★☆" if ("死叉" in r or "强烈" in r) else "★☆☆"
                lines.append(f"  ❌ {r.strip()} {star}")
    # 补充负面评分因素（避免只看到正面信号而忽略风险）
    neg_factors = []
    if macd_score <= -5:
        neg_factors.append(f"MACD空头运行（{macd_score}分），短期承压")
    if kdj_score <= -5:
        if kdj_k > kdj_d:  # V5.18.2: K在D之上=金叉≠死叉，修正标签
            neg_factors.append(f"KDJ超买区（{kdj_score}分），金叉但指标高企有回调风险")
        else:
            neg_factors.append(f"KDJ死叉（{kdj_score}分），动能转弱")
    if rsi_score < 0:
        if rsi > rsi_prev:  # V5.18.2: RSI仍在上升，"走弱"不准确
            neg_factors.append(f"RSI高企（{rsi_score}分），超买区域需谨慎")
        else:
            neg_factors.append(f"RSI走弱（{rsi_score}分），上行动能不足")
    if adx_score < 0:
        neg_factors.append(f"ADX空头趋势（{adx_score}分）")
    if ma_score < 0:
        neg_factors.append(f"均线空头排列（{ma_score}分）")
    if vol_score < 0:
        neg_factors.append(f"成交量异常（{vol_score}分）")
    if boll_score < 0:
        neg_factors.append(f"布林带压力（{boll_score}分）")
    if div_score < 0:
        neg_factors.append(f"RSI背离（{div_score}分），反转风险")
    if cp_score < 0:
        neg_factors.append(f"K线看空形态（{cp_score}分）")
    if vd_score < 0:
        neg_factors.append(f"量价背离（{vd_score}分），趋势不稳固")
    for nf in neg_factors:
        lines.append(f"  ⚠️ {nf}")
    lines.append("")
    # V5.14: 多周期一致性
    if weekly_trend and monthly_trend:
        lines.append("=" * 40)
        lines.append("📅 多周期一致性")
        lines.append("=" * 40)
        week_label = weekly_trend.get("label", "—")
        month_label = monthly_trend.get("label", "—")
        week_rsi = weekly_trend.get("rsi", "—")
        month_rsi = monthly_trend.get("rsi", "—")

        # 周期一致性评分
        periods = [
            ("日线", signal_cn, rsi),
            ("周线", week_label, week_rsi),
            ("月线", month_label, month_rsi),
        ]
        trends = [p[1] for p in periods]
        # 判断一致性：三线同向 = 共振
        bullish_count = sum(1 for t in trends if "买" in t or "多" in t)
        bearish_count = sum(1 for t in trends if "卖" in t or "空" in t)
        if mt_discount == 10:
            if bullish_count == 3:
                mt_label = "🔥 三线共振看多 — 信号置信度极高"
            else:
                mt_label = "⚠️ 三线共振看空 — 信号置信度极高"
            mt_bonus = "（+10分已计入总分）"
        elif mt_discount == -5:
            mt_label = "⚠️ 三线分歧 — 日线信号可能为假突破，建议观望"
            mt_bonus = "（-5分已计入总分）"
        elif bullish_count == 2:
            mt_label = "✅ 双线看多，日线信号可信度较高"
            mt_bonus = ""
        elif bearish_count == 2:
            mt_label = "⚠️ 双线看空，日线信号需谨慎"
            mt_bonus = ""
        else:
            mt_label = "⚠️ 三线分歧 — 日线信号可能为假突破，建议观望"
            mt_bonus = ""

        lines.append(f"  {'':4}日线：{signal_cn}（RSI={rsi}）")
        lines.append(f"  {'':4}周线：{week_label}（RSI={week_rsi}，{weekly_trend.get('price_vs_ma20','—')}）")
        lines.append(f"  {'':4}月线：{month_label}（RSI={month_rsi}，{monthly_trend.get('price_vs_ma20','—')}）")
        lines.append(f"  → {mt_label} {mt_bonus}")
        lines.append("")
    # 技术面详情
    lines.append("技术面详情")
    lines.append(f"  • RSI：{rsi}（{rsi_zone}）{rsi_arrow}较前日")
    macd_status = {"golden": "金叉", "death": "死叉"}.get(macd_cross, "多头运行" if macd_val > macd_signal else "空头运行")
    lines.append(f"  • MACD：{macd_status}，柱状图={macd_hist}")
    lines.append(f"  • KDJ：K={kdj_k}，D={kdj_d}（{kdj_zone}），K与D：{kdj_cross}")
    adx_desc = {"strong_bull": "强势多头", "weak_bull": "偏多", "strong_bear": "强势空头", "weak_bear": "偏空", "ranging": "震荡"}.get(adx_trend, "震荡")
    lines.append(f"  • ADX：{adx}（{adx_desc}），+DI={plus_di} / -DI={minus_di}")
    vol_cn = {"high_volume": "放量", "above_avg": "量能偏高", "low_volume": "缩量", "normal": "正常"}.get(volume_signal, "正常")
    lines.append(f"  • 成交量：{vol_cn}（比率{volume_ratio}x）")
    trend_cn = {"bullish": "偏多", "bearish": "偏空", "neutral": "震荡"}.get(trend_direction, "震荡")
    # V5.17.8: 与过滤后的 signal_cn 对齐（ADX<25 震荡市等场景）
    if "观望" in signal_cn:
        if trend_direction == "bullish":
            trend_cn = "偏多（"+("ADX<25 趋势不明" if adx_filtered else "但信号观望")+"，需确认）"
        elif trend_direction == "bearish":
            trend_cn = "偏空（"+("ADX<25 趋势不明" if adx_filtered else "但信号观望")+"，需确认）"
        else:
            trend_cn = "震荡（无明显方向）"
    lines.append(f"  • 趋势方向：{trend_cn}")
    # V5.16: ATR波动率
    if atr:
        atr_pct = round(atr / current_price * 100, 2) if current_price else 0
        lines.append(f"  • ATR(14)：{currency_symbol}{atr}（日波动 {atr_pct}%）")
    # 支撑阻力（百分比）
    if support_level and support_level != 0 and current_price != 0:
        sup_pct = round((support_level - current_price) / current_price * 100, 1)
        lines.append(f"  • 支撑位：{currency_symbol}{support_level}（距当前 {sup_pct:+}%）")
    if resistance_level and resistance_level != 0 and current_price != 0:
        res_pct = round((resistance_level - current_price) / current_price * 100, 1)
        lines.append(f"  • 阻力位：{currency_symbol}{resistance_level}（距当前 {res_pct:+}%）")
    lines.append("")
    # 操作建议
    lines.append("操作建议")
    # V5.18.1: RSI 超买(>80)或超卖(<20)时，操作建议提示风险
    rsi_risk = rsi and (rsi > 80 or rsi < 20)
    rsi_risk_note = "⚠️ RSI极度超买，回调风险高，建议等待回调" if rsi and rsi > 80 else ("⚠️ RSI极度超卖，反弹风险高，建议等待确认" if rsi and rsi < 20 else "")

    if signal in ("strong_buy", "buy"):
        lines.append(f"  - 保守策略：等待回调至支撑位 {currency_symbol}{support_level if support_level else '—'} 附近再入场")
        lines.append(f"  - 稳健策略：按入场价 {entry_str} 分批建仓，止损设 {stop_str}")
        lines.append(f"  - 激进策略：现价 {currency_symbol}{current_price} 直接入场，目标 {take_str}")
        if rsi_risk_note:
            lines.append(f"  {rsi_risk_note}")
    elif signal in ("strong_sell", "sell"):
        lines.append(f"  - 保守策略：继续持有观察，等待反弹至阻力位 {currency_symbol}{resistance_level if resistance_level else '—'} 再减仓")
        lines.append(f"  - 稳健策略：按当前价 {currency_symbol}{current_price} 分批减仓，止损设 {stop_str}")
        lines.append(f"  - 激进策略：现价直接清仓，等待下次买入信号")
        if rsi_risk_note:
            lines.append(f"  {rsi_risk_note}")
    else:
        if adx_filtered:
            lines.append(f"  - 保守策略：观望为主，等待 ADX>25（趋势明确）后再参考技术信号")
            lines.append(f"  - 稳健策略：暂不操作，关注 ADX 回升至 25 以上再考虑入场")
            lines.append(f"  - 激进策略：震荡市不建议操作，等 ADX>25 确认趋势后再行动")
        else:
            lines.append(f"  - 保守策略：观望为主，等待明确信号")
            lines.append(f"  - 稳健策略：暂不操作，等待指标进一步确认")
            lines.append(f"  - 激进策略：若突破 {currency_symbol}{resistance_level if resistance_level else '—'} 可少量试探")
    lines.append("")
    # V5.17.6: 基本面摘要
    if fundamentals and any(v is not None for v in fundamentals.values()):
        lines.append("=" * 40)
        lines.append("📋 基本面速览")
        lines.append("=" * 40)
        if fundamentals.get("pe"):
            lines.append(f"  • 市盈率(TTM)：{fundamentals['pe']}")
        if fundamentals.get("forward_pe"):
            lines.append(f"  • 远期市盈率：{fundamentals['forward_pe']}")
        if fundamentals.get("market_cap"):
            mcap = fundamentals["market_cap"]
            if mcap >= 1e12:
                mcap_str = f"{mcap/1e12:.2f}万亿"
            elif mcap >= 1e8:
                mcap_str = f"{mcap/1e8:.0f}亿"
            else:
                mcap_str = f"{mcap/1e6:.0f}百万"
            lines.append(f"  • 市值：{currency_symbol}{mcap_str}")
        if fundamentals.get("sector") or fundamentals.get("industry"):
            sector = fundamentals.get("sector", "")
            industry = fundamentals.get("industry", "")
            if sector and industry:
                lines.append(f"  • 行业：{sector} / {industry}")
            elif sector:
                lines.append(f"  • 行业：{sector}")
        if fundamentals.get("beta"):
            beta = fundamentals["beta"]
            beta_label = "高波动" if beta > 1.5 else ("中等波动" if beta > 1.0 else "低波动")
            lines.append(f"  • Beta：{beta}（{beta_label}）")
        if fundamentals.get("52w_high") and fundamentals.get("52w_low"):
            h52 = fundamentals["52w_high"]
            l52 = fundamentals["52w_low"]
            pos = round((current_price - l52) / (h52 - l52) * 100, 0) if h52 != l52 else 50
            lines.append(f"  • 52周范围：{currency_symbol}{l52} — {currency_symbol}{h52}（当前处于 {pos:.0f}% 分位）")
        if fundamentals.get("dividend_yield"):
            lines.append(f"  • 股息率：{fundamentals['dividend_yield']}%")
        lines.append("")
    lines.append("=" * 40)
    lines.append("以上分析基于技术指标客观数据，仅供个人投资参考。股市有风险，投资需谨慎。")
    lines.append("💡 跨资产参考：加密货币 BTC/USDT 和汇率 USD/JPY 可作为市场情绪的辅助验证指标。")
    if accuracy_data and accuracy_data.get("testable_days", 0) > 0:
        acc = accuracy_data.get("accuracy")
        days = accuracy_data.get("testable_days", 0)
        if acc is not None:
            lines.append(f"📈 方向一致率回测：近{days}个有效信号日，信号方向与实际走势一致率 {acc}%（{accuracy_data.get('consistent_days', 0)}/{days}）")
            lines.append("💡 技术分析靠 R:R 不对称获利（小亏大赚），不追求高胜率。一只大赚可覆盖多笔小亏。")
        elif accuracy_data.get("note"):
            lines.append(f"📈 方向一致率回测：{accuracy_data['note']}")
    elif accuracy_data and accuracy_data.get("note"):
        lines.append(f"📈 方向一致率回测：{accuracy_data['note']}")
    lines.append("=" * 40)

    # V5.17.5: 返回评分明细供 API 端点使用
    score_breakdown = {
        "rsi_divergence": div_score,
        "candle_pattern": cp_score,
        "volume_divergence": vd_score,
        "adx": adx_score,
        "macd": macd_score,
        "kdj": kdj_score,
        "rsi": rsi_score,
        "ma": ma_score,
        "volume": vol_score,
        "bollinger": boll_score,
    }
    return "\n".join(lines), {"total_score": int(total_score), "score_breakdown": score_breakdown}


def build_tradepoint_report(
    name, symbol, market, currency,
    current_price, change_percent,
    trade_point, trade_point_cn, score,
    buy_reasons_text, sell_reasons_text,
    entry_price, stop_loss, take_profit,
    rsi, rsi_prev,
    macd_cross, macd_hist,
    kdj_k, kdj_d, kdj_j,
    volume_ratio, trend_direction,
    support_level, resistance_level,
):
    """
    V5.11: API层预渲染单股买卖点分析报告。
    专精于入场/离场时机判断，输出买卖点触发原因和价格建议。
    """
    lines = []
    lines.append(f"📊 {name}（{symbol}）买卖点分析")
    lines.append("")

    change_sign = "+" if change_percent >= 0 else ""
    currency_symbol = "HK$" if currency == "HKD" else ("$" if currency == "USD" else "¥")
    lines.append(f"💰 当前价格：{currency_symbol}{current_price}（{change_sign}{change_percent}%）")

    tp_emoji_map = {
        "strong_buy": "🟢",
        "buy": "🟢",
        "strong_sell": "🔴",
        "sell": "🔴",
        "hold": "⚪",
    }
    tp_emoji = tp_emoji_map.get(trade_point, "⚪")
    lines.append(f"🎯 买卖点类型：{tp_emoji} {trade_point_cn}")
    lines.append(f"📊 综合评分：{score:+}")
    lines.append("=" * 40)

    # 买入理由
    lines.append("✅ 买入理由")
    if buy_reasons_text and buy_reasons_text != "暂无买入理由":
        for r in buy_reasons_text.split("；"):
            r = r.strip()
            if r:
                lines.append(f"  - {r}")
    else:
        lines.append("  暂无明确的买入信号")
    lines.append("")

    # 卖出理由
    lines.append("❌ 卖出理由")
    if sell_reasons_text and sell_reasons_text != "暂无卖出理由":
        for r in sell_reasons_text.split("；"):
            r = r.strip()
            if r:
                lines.append(f"  - {r}")
    else:
        lines.append("  暂无明确的卖出信号")
    lines.append("")

    # 价格建议
    lines.append("💰 价格建议")
    entry_str = f"{currency_symbol}{entry_price}" if entry_price and entry_price != 0 else "—"
    stop_str = f"{currency_symbol}{stop_loss}" if stop_loss and stop_loss != 0 else "—"
    take_str = f"{currency_symbol}{take_profit}" if take_profit and take_profit != 0 else "—"
    lines.append(f"  入场价：{entry_str}")
    lines.append(f"  止损价：{stop_str}")
    lines.append(f"  止盈价：{take_str}")
    lines.append("")

    # 核心指标
    lines.append("📊 核心指标")
    rsi_zone = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "正常")
    rsi_arrow = "▲" if rsi > rsi_prev else ("▼" if rsi < rsi_prev else "▬")
    lines.append(f"  • RSI：{rsi}（{rsi_zone}）{rsi_arrow}较前日")

    macd_status = {"golden": "金叉", "death": "死叉"}.get(macd_cross, "多头运行" if macd_hist > 0 else "空头运行")
    lines.append(f"  • MACD：{macd_status}，柱状图={macd_hist}")

    kdj_zone = "超卖区" if kdj_k < 20 else ("超买区" if kdj_k > 80 else "中性")
    kdj_cross = "金叉" if kdj_k > kdj_d else "死叉"
    lines.append(f"  • KDJ：K={kdj_k}，D={kdj_d}（{kdj_zone}），K与D：{kdj_cross}")

    vol_cn = {"high_volume": "放量", "above_avg": "量能偏高", "low_volume": "缩量", "normal": "正常"}.get(volume_ratio, f"{volume_ratio}x")
    lines.append(f"  • 成交量比率：{vol_cn}")
    trend_cn = {"bullish": "偏多", "bearish": "偏空", "neutral": "震荡"}.get(trend_direction, "震荡")
    # 与 trade_point 对齐：观望时趋势描述需保守
    if trade_point.lower() in ("hold",):
        if trend_direction == "bullish":
            trend_cn = "偏多（但信号观望，需确认）"
        elif trend_direction == "bearish":
            trend_cn = "偏空（但信号观望，需确认）"
        else:
            trend_cn = "震荡（无明显方向）"
    lines.append(f"  • 趋势方向：{trend_cn}")

    if support_level and support_level != 0 and current_price != 0:
        sup_pct = round((support_level - current_price) / current_price * 100, 1)
        lines.append(f"  • 支撑位：{currency_symbol}{support_level}（距当前 {sup_pct:+}%）")
    if resistance_level and resistance_level != 0 and current_price != 0:
        res_pct = round((resistance_level - current_price) / current_price * 100, 1)
        lines.append(f"  • 阻力位：{currency_symbol}{resistance_level}（距当前 {res_pct:+}%）")

    lines.append("")
    lines.append("=" * 40)
    lines.append("以上分析基于技术指标客观数据，仅供个人投资参考。股市有风险，投资需谨慎。")
    lines.append("💡 跨资产参考：加密货币 BTC/USDT 和汇率 USD/JPY 可作为市场情绪的辅助验证指标。")
    lines.append("=" * 40)

    return "\n".join(lines)


def build_crypto_report(
    name, symbol,
    current_price, change_percent,
    signal, confidence,
    signals_text,
    rsi, rsi_prev, rsi_delta,
    macd_val, macd_signal_val, macd_hist, macd_cross,
    kdj_k, kdj_d, kdj_j,
    adx, adx_trend, plus_di, minus_di,
    boll_upper, boll_middle, boll_lower,
    ma5, ma10, ma20, ma50,
    volume_signal, volume_ratio,
    support_level, resistance_level,
    kline_text,
    rsi_div_type, rsi_div_desc,
):
    """
    V5.11: API层预渲染加密货币分析报告。
    格式简化版（无买卖点/评分系统），突出核心指标和操作建议。
    """
    lines = []
    lines.append(f"📊 {name}（{symbol}）加密货币分析报告")
    lines.append("")

    change_sign = "+" if change_percent >= 0 else ""
    lines.append(f"💰 当前价格：${current_price}（{change_sign}{change_percent}%）")

    signal_cn_map = {
        "strong_buy": "强烈看多", "buy": "看多",
        "hold": "观望", "sell": "看空", "strong_sell": "强烈看空",
    }
    signal_cn = signal_cn_map.get(signal.lower() if signal else "", "观望")
    # 星级：基于买卖评分简单映射
    buy_score = 0
    sell_score = 0
    # 从信号文本粗略估算
    s_text = signals_text.lower() if signals_text else ""
    if "buy" in s_text or "看多" in s_text:
        buy_score = 5
    if "sell" in s_text or "看空" in s_text:
        sell_score = 5
    if adx_trend == "strong_bull":
        buy_score += 3
    elif adx_trend == "strong_bear":
        sell_score += 3

    if buy_score - sell_score >= 5:
        stars = "★★★"
    elif buy_score > sell_score:
        stars = "★★☆"
    elif sell_score > buy_score:
        stars = "★☆☆"
    else:
        stars = "★☆☆"
    conf_cn = {"high": "强", "medium": "中等", "low": "弱"}.get(confidence.lower() if confidence else "medium", "中等")
    lines.append(f"📌 综合信号：{signal_cn} {stars}（{conf_cn}）")
    lines.append("=" * 40)

    # RSI背离 + 极端警告
    if rsi_div_type and rsi_div_type != "none":
        lines.append(f"⚠️ RSI背离信号：{rsi_div_desc}")
    if rsi > 85:
        lines.append(f"⚠️ 检测到极端超买指标（RSI {rsi}），建议等待信号冷却后再操作。")
    elif rsi < 15:
        lines.append(f"⚠️ 检测到极端超卖指标（RSI {rsi}），建议等待信号冷却后再操作。")
    lines.append("")

    # 核心指标
    lines.append("📊 核心指标")
    rsi_zone = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "正常")
    rsi_arrow = "▲" if rsi > rsi_prev else ("▼" if rsi < rsi_prev else "▬")
    lines.append(f"  • RSI：{rsi}（{rsi_zone}）{rsi_arrow}较前日")

    macd_status = {"golden": "金叉", "death": "死叉"}.get(macd_cross, "多头运行" if macd_val > macd_signal_val else "空头运行")
    lines.append(f"  • MACD：{macd_status}，柱状图={macd_hist}")

    kdj_zone = "超卖区" if kdj_k < 20 else ("超买区" if kdj_k > 80 else "中性")
    kdj_cross = "金叉" if kdj_k > kdj_d else "死叉"
    lines.append(f"  • KDJ：K={kdj_k}，D={kdj_d}（{kdj_zone}），K与D：{kdj_cross}")

    adx_desc = {"strong_bull": "强势多头", "weak_bull": "偏多", "strong_bear": "强势空头", "weak_bear": "偏空", "ranging": "震荡"}.get(adx_trend, "震荡")
    lines.append(f"  • ADX：{adx}（{adx_desc}），+DI={plus_di} / -DI={minus_di}")

    ma_label = "多头排列" if ma5 and ma10 and ma20 and ma5 > ma10 > ma20 else ("空头排列" if ma5 and ma10 and ma20 and ma5 < ma10 < ma20 else "混杂")
    lines.append(f"  • 均线：{ma_label}")

    vol_cn = {"high_volume": "放量", "above_avg": "量能偏高", "low_volume": "缩量", "normal": "正常"}.get(volume_signal, "正常")
    lines.append(f"  • 成交量：{vol_cn}（比率{volume_ratio}x）")

    if boll_lower and boll_upper:
        boll_pos = "靠近上轨" if current_price >= boll_upper * 0.99 else ("靠近下轨" if current_price <= boll_lower * 1.01 else "中轨附近")
        lines.append(f"  • 布林带：{boll_pos}（上轨{boll_upper}，下轨{boll_lower}）")

    lines.append("")

    # 核心信号汇总
    if signals_text:
        lines.append("核心信号汇总")
        for s in signals_text.split("；"):
            s = s.strip()
            if s:
                if "金叉" in s or "背离" in s or "超卖" in s:
                    lines.append(f"  - {s} ★★★")
                elif "死叉" in s or "超买" in s:
                    lines.append(f"  - {s} ★☆☆")
                else:
                    lines.append(f"  - {s} ★★☆")
        lines.append("")

    # 近期K线
    lines.append("近期K线")
    if kline_text:
        for line in kline_text.split("\n"):
            if line.strip():
                lines.append(f"  {line.strip()}")
    lines.append("")

    # 操作建议
    lines.append("操作建议")
    if signal.lower() in ("strong_buy", "buy"):
        lines.append(f"  - 保守策略：等待回调至支撑位 ${support_level if support_level else '—'} 附近再入场")
        lines.append(f"  - 稳健策略：分批建仓，控制仓位不超过总资金的20%")
        lines.append(f"  - 激进策略：现价 ${current_price} 直接入场")
    elif signal.lower() in ("strong_sell", "sell"):
        lines.append(f"  - 保守策略：继续持有观察，等待反弹至阻力位 ${resistance_level if resistance_level else '—'} 再减仓")
        lines.append(f"  - 稳健策略：按当前价 ${current_price} 分批减仓")
        lines.append(f"  - 激进策略：现价直接清仓，等待下次买入信号")
    else:
        lines.append(f"  - 保守策略：观望为主，等待明确信号")
        lines.append(f"  - 稳健策略：可小仓位试探")
        lines.append(f"  - 激进策略：短线操作者可在支撑位附近抢反弹")
    lines.append("")
    lines.append("=" * 40)
    lines.append("以上分析基于技术指标客观数据，仅供个人投资参考。加密货币波动剧烈，投资需谨慎。")

    return "\n".join(lines)


def build_forex_report(
    pair, name,
    current_price, change_percent,
    signal, confidence,
    signals_text,
    rsi, rsi_prev, rsi_delta,
    macd_val, macd_signal_val, macd_hist, macd_cross,
    kdj_k, kdj_d, kdj_j,
    adx, adx_trend, plus_di, minus_di,
    boll_upper, boll_middle, boll_lower,
    ma5, ma10, ma20, ma50,
    volume_signal, volume_ratio,
    support_level, resistance_level,
    kline_text,
    rsi_div_type, rsi_div_desc,
    volatility_20d,
):
    """
    V5.11: API层预渲染汇率分析报告。
    格式简洁版，突出汇率特有指标（波动率）。
    """
    lines = []
    lines.append(f"📊 {name}（{pair}）汇率分析报告")
    lines.append("")

    change_sign = "+" if change_percent >= 0 else ""
    lines.append(f"💰 当前汇率：{current_price}（{change_sign}{change_percent}%）")

    signal_cn_map = {
        "strong_buy": "强烈看多本币", "buy": "看多本币",
        "hold": "观望", "sell": "看空本币", "strong_sell": "强烈看空本币",
    }
    signal_cn = signal_cn_map.get(signal.lower() if signal else "", "观望")
    conf_cn = {"high": "强", "medium": "中等", "low": "弱"}.get(confidence.lower() if confidence else "medium", "中等")
    lines.append(f"📌 综合信号：{signal_cn}（{conf_cn}）")

    if volatility_20d:
        lines.append(f"📈 20日波动率：{volatility_20d}%")
    lines.append("=" * 40)

    if rsi_div_type and rsi_div_type != "none":
        lines.append(f"⚠️ RSI背离信号：{rsi_div_desc}")
    if rsi > 85:
        lines.append(f"⚠️ 检测到极端超买指标（RSI {rsi}），汇率可能回调。")
    elif rsi < 15:
        lines.append(f"⚠️ 检测到极端超卖指标（RSI {rsi}），汇率可能反弹。")
    lines.append("")

    lines.append("📊 核心指标")
    rsi_zone = "超买" if rsi > 70 else ("超卖" if rsi < 30 else "正常")
    rsi_arrow = "▲" if rsi > rsi_prev else ("▼" if rsi < rsi_prev else "▬")
    lines.append(f"  • RSI：{rsi}（{rsi_zone}）{rsi_arrow}较前日")

    macd_status = {"golden": "金叉", "death": "死叉"}.get(macd_cross, "多头运行" if macd_val > macd_signal_val else "空头运行")
    lines.append(f"  • MACD：{macd_status}，柱状图={macd_hist}")

    kdj_zone = "超卖区" if kdj_k < 20 else ("超买区" if kdj_k > 80 else "中性")
    kdj_cross = "金叉" if kdj_k > kdj_d else "死叉"
    lines.append(f"  • KDJ：K={kdj_k}，D={kdj_d}（{kdj_zone}），K与D：{kdj_cross}")

    adx_desc = {"strong_bull": "强势", "weak_bull": "偏多", "strong_bear": "弱势", "weak_bear": "偏空", "ranging": "震荡"}.get(adx_trend, "震荡")
    lines.append(f"  • ADX趋势强度：{adx}（{adx_desc}）")

    vol_cn = {"high_volume": "放量", "above_avg": "量能偏高", "low_volume": "缩量", "normal": "正常"}.get(volume_signal, "正常")
    lines.append(f"  • 成交量：{vol_cn}")
    lines.append("")

    if signals_text:
        lines.append("核心信号汇总")
        for s in signals_text.split("；"):
            s = s.strip()
            if s:
                lines.append(f"  - {s}")
        lines.append("")

    lines.append("近期K线")
    if kline_text:
        for line in kline_text.split("\n"):
            if line.strip():
                lines.append(f"  {line.strip()}")
    lines.append("")

    lines.append("操作建议")
    if signal.lower() in ("strong_buy", "buy"):
        lines.append(f"  - 保守策略：等待回调至 {support_level if support_level else '近期低位'} 附近再购汇")
        lines.append(f"  - 稳健策略：分批购汇，控制汇率波动风险")
        lines.append(f"  - 激进策略：现价 {current_price} 直接购汇")
    elif signal.lower() in ("strong_sell", "sell"):
        lines.append(f"  - 保守策略：继续观察，等待反弹至 {resistance_level if resistance_level else '近期高位'} 再结汇")
        lines.append(f"  - 稳健策略：分批结汇")
        lines.append(f"  - 激进策略：现价直接结汇")
    else:
        lines.append(f"  - 保守策略：观望为主，等待明确趋势")
        lines.append(f"  - 稳健策略：可小金额试探性操作")
        lines.append(f"  - 激进策略：短线操作者可在支撑位附近抢反弹")
    lines.append("")
    lines.append("=" * 40)
    lines.append("以上分析基于技术指标客观数据，仅供个人参考。汇率波动受多重因素影响，投资需谨慎。")

    return "\n".join(lines)


def build_scan_report(
    market, total_scanned, total_signals, failed_count,
    buy_stocks, sell_stocks,
    divergence_stocks,
    scan_time,
):
    """
    V5.11: API层预渲染批量扫描报告。
    输出买入/卖出信号列表、RSI背离预警、扫描摘要。
    """
    lines = []
    market_cn = {"us": "美股", "hk": "港股", "cn": "A股"}.get(market, "美港股")
    lines.append(f"📊 批量扫描报告（{market_cn}）")
    lines.append(f"⏰ 扫描时间：{scan_time[:19] if scan_time else 'N/A'}")
    lines.append(f"🔍 扫描结果：扫描{total_scanned}只 → 发现{total_signals}个信号")
    if failed_count > 0:
        lines.append(f"⚠️ 失败：{failed_count}只（数据获取异常）")
    lines.append("=" * 40)

    # 买入信号
    buy_count = len(buy_stocks)
    lines.append(f"🟢 买入信号（{buy_count}个）")
    if buy_count > 0:
        currency_map = {"us": "$", "hk": "HK$", "cn": "¥"}
        cs = currency_map.get(market, "$")
        for i, s in enumerate(buy_stocks, 1):
            stars = "★★★" if abs(s.get("score", 0)) >= 7 else "★★☆"
            tp_cn = {"strong_buy": "强烈买入", "buy": "建议买入"}.get(s.get("trade_point", ""), "建议买入")
            entry = f"{cs}{s.get('entry_price', '—')}" if s.get('entry_price') and s.get('entry_price') != 0 else "—"
            stop = f"{cs}{s.get('stop_loss', '—')}" if s.get('stop_loss') and s.get('stop_loss') != 0 else "—"
            take = f"{cs}{s.get('take_profit', '—')}" if s.get('take_profit') and s.get('take_profit') != 0 else "—"
            lines.append(f"{i}. {s.get('symbol', '?')} — {tp_cn} {stars} — 评分{s.get('score', 0):+}")
            lines.append(f"   入场{entry} / 止损{stop} / 止盈{take}")
            reasons = s.get("buy_reasons_text", "")
            if reasons:
                # 取前3条理由
                reason_list = [r.strip() for r in reasons.split("；") if r.strip()]
                for r in reason_list[:3]:
                    lines.append(f"   → {r}")
        lines.append("")
    else:
        lines.append("  无")
        lines.append("")

    # 卖出信号
    sell_count = len(sell_stocks)
    lines.append(f"🔴 卖出信号（{sell_count}个）")
    if sell_count > 0:
        currency_map = {"us": "$", "hk": "HK$", "cn": "¥"}
        cs = currency_map.get(market, "$")
        for i, s in enumerate(sell_stocks, 1):
            stars = "★★★" if abs(s.get("score", 0)) >= 7 else "★☆☆"
            tp_cn_map = {"strong_sell": "强烈卖出", "sell": "建议卖出"}
            tp_cn = tp_cn_map.get(s.get("trade_point", ""), "建议卖出")
            entry = f"{cs}{s.get('entry_price', '—')}" if s.get('entry_price') and s.get('entry_price') != 0 else "—"
            stop = f"{cs}{s.get('stop_loss', '—')}" if s.get('stop_loss') and s.get('stop_loss') != 0 else "—"
            take = f"{cs}{s.get('take_profit', '—')}" if s.get('take_profit') and s.get('take_profit') != 0 else "—"
            lines.append(f"{i}. {s.get('symbol', '?')} — {tp_cn} {stars} — 评分{s.get('score', 0):+}")
            lines.append(f"   出场{entry} / 反弹止损{stop} / 止盈{take}")
            reasons = s.get("sell_reasons_text", "")
            if reasons:
                reason_list = [r.strip() for r in reasons.split("；") if r.strip()]
                for r in reason_list[:3]:
                    lines.append(f"   → {r}")
        lines.append("")
    else:
        lines.append("  无")
        lines.append("")

    # RSI背离预警
    div_count = len(divergence_stocks)
    lines.append(f"⚠️ RSI背离预警（{div_count}只）")
    if div_count > 0:
        for d in divergence_stocks:
            div_type = d.get("type", "none")
            desc = d.get("desc", "")
            emoji = "🔴" if "bearish" in str(div_type) else "🟢"
            lines.append(f"  {emoji} {d.get('symbol', '?')} — {desc}")
    else:
        lines.append("  无")
    lines.append("")

    lines.append("=" * 40)
    lines.append("以上分析基于技术指标客观数据，仅供个人投资参考。股市有风险，投资需谨慎。")
    lines.append("💡 跨资产参考：加密货币 BTC/USDT 和汇率 USD/JPY 可作为市场情绪的辅助验证指标。")

    return "\n".join(lines)


def build_compare_report(
    market, total, success,
    stocks, summary,
):
    """
    V5.12: API层预渲染多股对比分析报告。
    输入：单股分析结果列表（stocks）+ 对比摘要字典（summary）
    输出：完整中文对比报告，含对比表格和维度总结。
    """
    lines = []
    market_cn = {"us": "美股", "hk": "港股", "cn": "A股"}.get(market, "美港股")
    currency_map = {"us": "$", "hk": "HK$", "cn": "¥"}
    cs = currency_map.get(market, "$")

    lines.append(f"📊 多股对比分析报告（{market_cn}）")
    lines.append(f"🔍 对比{total}只 → {success}只有效数据")
    lines.append("")

    # 信号映射
    signal_cn_map = {
        "strong_buy": "强烈买入", "buy": "建议买入",
        "hold": "观望", "sell": "建议卖出", "strong_sell": "强烈卖出",
    }

    # 对比表格
    lines.append("## 核心指标对比")
    lines.append("")
    lines.append("| 股票 | 现价 | 涨跌幅 | 信号 | RSI | MACD | KDJ-K | 量比 | ADX | ADX趋势 |")
    lines.append("|------|------|--------|------|-----|------|-------|------|-----|---------|")

    for s in stocks:
        name = s.get("name", s.get("symbol", "?"))
        symbol = s.get("symbol", "?")
        if s.get("status") == "ok":
            price = f"{cs}{s.get('current_price', '--')}"
            change = f"{s.get('change_percent', 0):+}%"
            sig = signal_cn_map.get(s.get("signal", "hold"), "观望")
            rsi = s.get("rsi", "--")
            macd_cross = {"golden": "金叉", "death": "死叉"}.get(s.get("macd_cross", "none"), "—")
            kdj_k = s.get("kdj_k", "--")
            vol = f"{s.get('volume_ratio', '--')}x"
            adx = s.get("adx", "--")
            adx_t = {"strong_bull": "强多", "weak_bull": "偏多", "strong_bear": "强空", "weak_bear": "偏空", "ranging": "震荡"}.get(s.get("adx_trend", ""), "—")
            lines.append(f"| {name}({symbol}) | {price} | {change} | {sig} | {rsi} | {macd_cross} | {kdj_k} | {vol} | {adx} | {adx_t} |")
        else:
            lines.append(f"| {name}({symbol}) | — | — | ❌数据异常 | — | — | — | — | — | — |")

    lines.append("")

    # 单股详细信号
    lines.append("## 单股信号详情")
    lines.append("")
    for s in stocks:
        if s.get("status") != "ok":
            continue
        name = s.get("name", s.get("symbol", "?"))
        symbol = s.get("symbol", "?")
        sig_cn = signal_cn_map.get(s.get("signal", "hold"), "观望")
        kdj_zone = "超卖区" if s.get("kdj_k", 50) < 20 else ("超买区" if s.get("kdj_k", 50) > 80 else "中性")
        rsi_zone = "超买" if s.get("rsi", 50) > 70 else ("超卖" if s.get("rsi", 50) < 30 else "正常")

        # RSI背离
        div_type = s.get("rsi_divergence_type", "none")
        div_desc = s.get("rsi_divergence_desc", "")
        div_warning = f" ⚠️{div_desc}" if div_type != "none" else ""

        lines.append(f"### {name}（{symbol}）")
        lines.append(f"- 信号：{sig_cn}{div_warning}")
        lines.append(f"- 价格：{cs}{s.get('current_price', '--')}（{s.get('change_percent', 0):+}%）")
        lines.append(f"- RSI：{s.get('rsi', '--')}（{rsi_zone}）")
        lines.append(f"- KDJ：K={s.get('kdj_k', '--')} / D={s.get('kdj_d', '--')}（{kdj_zone}）")
        lines.append(f"- MACD：{s.get('macd_histogram', 0):.4f}（{'多头运行' if s.get('macd_histogram', 0) > 0 else '空头运行'}）")
        lines.append(f"- ADX：{s.get('adx', '--')}（{'强势' if s.get('adx', 0) >= 25 else '弱势'}）")
        sup = f"{cs}{s.get('support_level', '—')}" if s.get('support_level') and s.get('support_level') != 0 else "—"
        res = f"{cs}{s.get('resistance_level', '—')}" if s.get('resistance_level') and s.get('resistance_level') != 0 else "—"
        lines.append(f"- 支撑：{sup} / 阻力：{res}")
        lines.append("")

    # RSI背离预警
    if summary.get("divergence_warnings"):
        lines.append("## ⚠️ RSI背离预警")
        lines.append("")
        valid_stocks = [s for s in stocks if s.get("status") == "ok" and s.get("rsi_divergence_type") != "none"]
        for s in valid_stocks:
            desc = s.get("rsi_divergence_desc", "")
            emoji = "🔴" if "bearish" in str(s.get("rsi_divergence_type", "")) else "🟢"
            lines.append(f"- {emoji} {s.get('name', s.get('symbol', '?'))}（{s.get('symbol', '?')}）：{desc}")
        lines.append("")

    # 对比总结
    lines.append("## 对比总结")
    lines.append("")
    if summary.get("best_performer"):
        lines.append(f"- 🏆 涨幅最强：{summary['best_performer']['symbol']}（{summary['best_performer']['change']:+}%）")
    if summary.get("worst_performer"):
        lines.append(f"- 📉 跌幅最大：{summary['worst_performer']['symbol']}（{summary['worst_performer']['change']:+}%）")
    if summary.get("rsi_highest"):
        sym = summary["rsi_highest"]["symbol"]
        rsi_v = summary["rsi_highest"]["rsi"]
        lines.append(f"- 🔥 RSI最高：{sym}（{rsi_v}，{'超买区' if rsi_v >= 70 else '正常'}）")
    if summary.get("rsi_lowest"):
        sym = summary["rsi_lowest"]["symbol"]
        rsi_v = summary["rsi_lowest"]["rsi"]
        lines.append(f"- 🧊 RSI最低：{sym}（{rsi_v}，{'超卖区' if rsi_v <= 30 else '正常'}）")
    if summary.get("strongest_trend"):
        lines.append(f"- 📈 趋势最强：{summary['strongest_trend']['symbol']}（ADX={summary['strongest_trend']['adx']}，{summary['strongest_trend']['adx_trend']}）")

    lines.append("")
    lines.append("=" * 40)
    lines.append("以上分析基于技术指标客观数据，仅供个人投资参考。投资有风险，入市需谨慎。")
    lines.append("💡 跨资产参考：加密货币 BTC/USDT 和汇率 USD/JPY 可作为市场情绪的辅助验证指标。")

    return "\n".join(lines)


def normalize_stock_symbol(symbol: str, market: str = "us") -> tuple:
    """
    标准化股票代码，自动补全市场后缀

    - 港股(hk)：纯数字代码自动补 .HK（如 0700 → 0700.HK, 00700 → 00700.HK）
    - 美股(us)：不做处理（yfinance 直接支持）
    - A股(cn)：补 .SS（上交所）或 .SZ（深交所）
    - auto：根据代码特征自动检测市场（纯数字3-5位→港股，6位纯数字→A股，其余→美股）
    - 已有后缀的代码直接返回

    returns: (normalized_symbol, detected_market)
    """
    sym = symbol.strip().upper()

    # 已有后缀，直接返回
    if sym.endswith(".HK") or sym.endswith(".SS") or sym.endswith(".SZ"):
        detected = "hk" if sym.endswith(".HK") else "cn"
        return sym, detected

    # auto 模式：根据代码特征自动检测市场
    if market.lower() == "auto":
        # 港股：纯数字 3-5 位
        if sym.isdigit() and 3 <= len(sym) <= 5:
            return f"{sym}.HK", "hk"
        # A股：6位纯数字，6开头→上交所(.SS)，0/3开头→深交所(.SZ)
        if sym.isdigit() and len(sym) == 6:
            if sym.startswith("6"):
                return f"{sym}.SS", "cn"
            elif sym.startswith("0") or sym.startswith("3"):
                return f"{sym}.SZ", "cn"
            else:
                return sym, "cn"  # 其他6位数字（如科创板688xxx）
        # 其余→美股（纯字母代码）
        return sym, "us"

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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 逐股检测，不提前转换
    detect_market = market.lower() if market else "us"

    # 如果没传 symbols，使用默认列表
    if not symbols or symbols.strip() == "":
        if detect_market == "hk":
            symbols = DEFAULT_HK_SCAN
        elif detect_market == "cn":
            symbols = DEFAULT_CN_SCAN
        else:
            symbols = DEFAULT_US_SCAN

    try:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(symbol_list) > 20:
            symbol_list = symbol_list[:20]

        # 标准化股票代码（auto 模式下逐股自动检测市场）
        normalized = [normalize_stock_symbol(s, detect_market) for s in symbol_list]

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
                    "adx": round(signal_data["indicators"]["adx"]["adx"], 2),
                    "adx_trend": str(signal_data["indicators"]["adx"]["trend"]),
                    "rsi_divergence_type": str(signal_data["indicators"]["rsi_divergence"]["type"]),
                    "rsi_divergence_desc": str(signal_data["indicators"]["rsi_divergence"]["description"]),
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
        # RSI背离股票（P0新增）
        divergence_stocks = [
            {"symbol": r["symbol"], "type": r["rsi_divergence_type"], "desc": r["rsi_divergence_desc"]}
            for r in results if r.get("rsi_divergence_type") != "none"
        ]

        scan_time = now_cn()
        return {
            "market": market,
            "total_scanned": len(symbol_list),
            "total_signals": len(results),
            "failed_count": failed_count,
            "last_error": last_error if failed_count > 0 else "",
            "buy_count": len(buy_stocks),
            "sell_count": len(sell_stocks),
            "divergence_count": len(divergence_stocks),
            "divergence_stocks": divergence_stocks,
            "top_buy": buy_stocks[:3] if buy_stocks else [],
            "top_sell": sell_stocks[:3] if sell_stocks else [],
            "all_signals": results,
            "scan_time": scan_time,
            # V5.11: API层预渲染批量扫描报告
            "formatted_report": build_scan_report(
                market=market,
                total_scanned=len(symbol_list),
                total_signals=len(results),
                failed_count=failed_count,
                buy_stocks=buy_stocks[:3] if buy_stocks else [],
                sell_stocks=sell_stocks[:3] if sell_stocks else [],
                divergence_stocks=divergence_stocks,
                scan_time=scan_time,
            ),
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
    # auto 模式：保留 auto 传给 normalize_stock_symbol 自动检测
    detect_market = market if market else "us"

    symbol, market = normalize_stock_symbol(symbol, detect_market)

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

        # V5.11: API层预渲染买卖点分析报告
        buy_reasons_text = "；".join(trade_points["buy_reasons"]) if trade_points["buy_reasons"] else "暂无买入理由"
        sell_reasons_text = "；".join(trade_points["sell_reasons"]) if trade_points["sell_reasons"] else "暂无卖出理由"
        formatted_report = build_tradepoint_report(
            name=str(info.get("longName", "N/A")),
            symbol=symbol,
            market=str(market),
            currency=str(info.get("currency", "USD")),
            current_price=current_price,
            change_percent=change_percent,
            trade_point=str(trade_points["trade_point"]),
            trade_point_cn=trade_point_cn.get(trade_points["trade_point"], "观望"),
            score=trade_points["score"],
            buy_reasons_text=buy_reasons_text,
            sell_reasons_text=sell_reasons_text,
            entry_price=trade_points["entry_price"],
            stop_loss=trade_points["stop_loss"],
            take_profit=trade_points["take_profit"],
            rsi=round(indicators["rsi"], 2),
            rsi_prev=round(indicators["rsi_prev"], 2),
            macd_cross="golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none"),
            macd_hist=round(indicators["macd"]["histogram"], 4),
            kdj_k=round(indicators["kdj"]["k"], 2),
            kdj_d=round(indicators["kdj"]["d"], 2),
            kdj_j=round(indicators["kdj"]["j"], 2),
            volume_ratio=signal_data["volume_ratio"],
            trend_direction=str(signal_data["trend_direction"]),
            support_level=signal_data["support_level"],
            resistance_level=signal_data["resistance_level"],
        )

        return {
            "symbol": str(symbol),
            "name": str(info.get("longName", "N/A")),
            "current_price": current_price,
            "change_percent": change_percent,
            "currency": str(info.get("currency", "USD")),
            "market": str(market),
            "analysis_time": now_cn(),
            # V5.11: 预渲染报告（Agent直接输出）
            "formatted_report": formatted_report,
            "trade_point": str(trade_points["trade_point"]),
            "trade_point_cn": trade_point_cn.get(trade_points["trade_point"], "观望"),
            "score": trade_points["score"],
            "buy_reasons_text": buy_reasons_text,
            "sell_reasons_text": sell_reasons_text,
            "entry_price": trade_points["entry_price"],
            "stop_loss": trade_points["stop_loss"],
            "take_profit": trade_points["take_profit"],
            "rsi": round(indicators["rsi"], 2),
            "rsi_prev": round(indicators["rsi_prev"], 2),
            "macd_cross": str("golden" if indicators["macd"]["golden_cross"] else ("death" if indicators["macd"]["death_cross"] else "none")),
            "macd_histogram": round(indicators["macd"]["histogram"], 4),
            "kdj_k": round(indicators["kdj"]["k"], 2),
            "kdj_d": round(indicators["kdj"]["d"], 2),
            "kdj_j": round(indicators["kdj"]["j"], 2),
            "volume_ratio": signal_data["volume_ratio"],
            "trend_direction": str(signal_data["trend_direction"]),
            "support_level": signal_data["support_level"],
            "resistance_level": signal_data["resistance_level"],
            "adx": round(indicators["adx"]["adx"], 2),
            "adx_trend": str(indicators["adx"]["trend"]),
            "rsi_divergence_type": str(indicators["rsi_divergence"]["type"]),
            "rsi_divergence_desc": str(indicators["rsi_divergence"]["description"]),
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

    dimension_count = (9 if hour_zhi else 8) if (birthday and b_shengke) else (8 if hour_zhi else 7)
    _dim_names = "旺行+生我行+日月+月相+六柱干支+纳音五行+飞星方位"
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
        parts = date_str.split('-')
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

    # ===== 第二步：历史统计加权 =====
    stat_periods = min(50, len(_SSQ_HISTORY))
    stat_data = _SSQ_HISTORY[:stat_periods]

    red_freq = {i: 0 for i in range(1, 34)}
    blue_freq = {i: 0 for i in range(1, 17)}
    red_miss = {i: 0 for i in range(1, 34)}
    blue_miss = {i: 0 for i in range(1, 17)}

    for rec in stat_data:
        for n in rec["red"]:
            red_freq[n] += 1
        blue_freq[rec["blue"]] += 1

    # 遗漏值
    for num in range(1, 34):
        for rec in stat_data:
            if num in rec["red"]:
                break
            red_miss[num] += 1
    for num in range(1, 17):
        for rec in stat_data:
            if num == rec["blue"]:
                break
            blue_miss[num] += 1

    # 统计分数（频率权重+遗漏权重）
    stat_red_score = {}
    stat_blue_score = {}
    max_rf = max(red_freq.values()) if red_freq.values() else 1
    max_rm = max(red_miss.values()) if red_miss.values() else 1
    max_bf = max(blue_freq.values()) if blue_freq.values() else 1
    max_bm = max(blue_miss.values()) if blue_miss.values() else 1

    for n in range(1, 34):
        # 频率归一化 + 遗漏归一化
        freq_score = red_freq[n] / max_rf  # 0-1
        miss_score = red_miss[n] / max_rm  # 0-1，遗漏越大越可能出
        stat_red_score[n] = freq_score * 0.6 + miss_score * 0.4  # 频率6:遗漏4

    for n in range(1, 17):
        freq_score = blue_freq[n] / max_bf
        miss_score = blue_miss[n] / max_bm
        stat_blue_score[n] = freq_score * 0.6 + miss_score * 0.4

    # ===== 第三步：融合评分 =====
    # v3.5: 冷门号保底分（玄学0分的号码给0.5基础分，避免完全排除）
    _COLD_FLOOR = 0.5
    final_red_score = {}
    for n in range(1, 34):
        x_score = xuanxue_red_score.get(n, 0)
        s_score = stat_red_score.get(n, 0) * 5  # 统计分归一化到0-5
        # 冷门号保底：玄学得分为0但有统计数据的号码，给基础分
        floor = _COLD_FLOOR if x_score == 0 and s_score > 0 else 0
        final_red_score[n] = x_score + s_score + floor

    final_blue_score = {}
    for n in range(1, 17):
        x_score = xuanxue_blue_score.get(n, 0)
        s_score = stat_blue_score.get(n, 0) * 5
        floor = _COLD_FLOOR if x_score == 0 and s_score > 0 else 0
        final_blue_score[n] = x_score + s_score + floor

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
    lines.append(f"统计权重：频率60% + 遗漏40%")
    _blue_weight_str = " / ".join(f"{k}×{v}" for k,v in _weights_blue.items() if v > 0)
    if _blue_weight_str != _weight_str:
        lines.append(f"蓝球独立权重：{_blue_weight_str}")
    if birth_weight_str:
        lines.append(f"🎂出生维度权重：{birth_weight_str}")
    lines.append(f"红球候选TOP18：{', '.join(f'{n:02d}({final_red_score[n]:.1f})' for n in red_pool)}")
    lines.append(f"蓝球候选TOP6：{', '.join(f'{n:02d}({final_blue_score[n]:.1f})' for n in blue_pool)}")
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
        "combinations": combinations,
        "red_pool_top18": red_pool,
        "blue_pool_top6": blue_pool,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
