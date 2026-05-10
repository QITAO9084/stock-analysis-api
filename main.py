"""
美股港股买卖点分析 API 服务
基于 yfinance + FastAPI
供 Coze 自定义插件接入
"""

import warnings
warnings.filterwarnings('ignore')

import time
import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI(
    title="Stock Analysis API",
    description="美股港股买卖点分析接口，供 Coze AI Agent 调用",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 简易请求缓存，避免同一符号短时间内重复拉取
_cache: dict[str, tuple[float, object]] = {}
CACHE_TTL = 300  # 5分钟缓存


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 1.3+ 返回 MultiIndex 列名，统一展平为单层"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # 去重列名
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values[1:]] = f"{dup}_"
    df.columns = cols.tolist()
    return df


def _fetch_yahoo_data(sym: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """底层：用 requests 直接请求 Yahoo Finance API（绕过 yfinance 的限速检测）"""
    try:
        import requests as req
        from datetime import datetime, timedelta

        # 计算 Unix 时间戳
        now = datetime.utcnow()
        period_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5d": 5}
        days = period_map.get(period, 180)
        start = int((now - timedelta(days=days)).timestamp())
        end = int(now.timestamp())

        interval_map = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}
        yf_interval = interval_map.get(interval, "1d")

        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
            f"?period1={start}&period2={end}&interval={yf_interval}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            return pd.DataFrame()

        r = result[0]
        timestamps = r.get("timestamp", [])
        quotes = r.get("indicators", {}).get("quote", [{}])[0]

        records = []
        for i, ts in enumerate(timestamps):
            row = {
                "Open": quotes.get("open", [None])[i],
                "High": quotes.get("high", [None])[i],
                "Low": quotes.get("low", [None])[i],
                "Close": quotes.get("close", [None])[i],
                "Volume": quotes.get("volume", [None])[i],
            }
            if row["Close"] is not None:
                records.append(row)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df.index = pd.to_datetime(timestamps[: len(records)], unit="s")
        df.index.name = "Date"
        return df
    except Exception:
        return pd.DataFrame()


def safe_download(sym: str, period: str = "6mo", interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    """带重试和限速保护的数据下载（yfinance → 直接请求 降级）"""
    cache_key = f"{sym}_{period}_{interval}"
    now = time.time()
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return data

    # 优先用 yfinance
    for attempt in range(max_retries):
        try:
            df = yf.download(sym, period=period, interval=interval, progress=False)
            df = flatten_columns(df)
            if not df.empty:
                _cache[cache_key] = (now, df)
                return df
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))

    # 降级：直接请求 Yahoo API
    time.sleep(2)
    df = _fetch_yahoo_data(sym, period, interval)
    if not df.empty:
        _cache[cache_key] = (now, df)
        return df

    return pd.DataFrame()


# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

def safe_info(sym: str, max_retries: int = 3) -> dict:
    """带重试和降级的 info 获取"""
    cache_key = f"{sym}_info"
    now = time.time()
    if cache_key in _cache:
        ts, data = _cache[cache_key]
        if now - ts < CACHE_TTL:
            return data

    # 优先 yfinance
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(sym)
            info = ticker.info
            if info:
                _cache[cache_key] = (now, info)
                return info
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(5 * (attempt + 1))

    # 降级：用 Yahoo API 获取基本信息
    try:
        import requests as req
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=5d&interval=1d"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = req.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("chart", {}).get("result", [{}])
        if data:
            meta = data[0].get("meta", {})
            info = {
                "shortName": meta.get("symbol", sym),
                "longName": meta.get("longName", sym),
                "regularMarketPrice": meta.get("regularMarketPrice"),
                "previousClose": meta.get("previousClose", {}).get("raw", meta.get("chartPreviousClose")),
                "fiftyTwoWeekHigh": meta.get("fiftyTwoWeekHigh"),
                "fiftyTwoWeekLow": meta.get("fiftyTwoWeekLow"),
                "currency": meta.get("currency", "USD"),
                "marketCap": meta.get("marketCap"),
            }
            _cache[cache_key] = (now, info)
            return info
    except Exception:
        pass

    return {}


def safe_history(sym: str, period: str = "5d", max_retries: int = 3) -> pd.DataFrame:
    """带重试的历史数据获取（降级到直接请求）"""
    return safe_download(sym, period=period, interval="1d", max_retries=max_retries)


def normalize_symbol(symbol: str, market: str = "auto") -> str:
    """标准化股票代码"""
    s = symbol.strip().upper()
    if market == "hk" and not s.endswith(".HK"):
        s = s + ".HK"
    return s


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def detect_macd_cross(hist: pd.Series) -> dict:
    """检测 MACD 金叉/死叉"""
    if len(hist) < 3:
        return {"status": "neutral", "desc": "数据不足"}
    last2 = hist.iloc[-2:].values
    prev2 = hist.iloc[-4:-2].values
    # 金叉：由负转正
    if last2[0] < 0 and last2[1] > 0:
        return {"status": "golden_cross", "desc": "金叉（买入信号）"}
    # 死叉：由正转负
    if last2[0] > 0 and last2[1] < 0:
        return {"status": "death_cross", "desc": "死叉（卖出信号）"}
    return {"status": "neutral", "desc": "中性"}


def find_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """简单支撑/压力位：近期最低/最高"""
    recent = df.tail(window)
    support = round(recent["Low"].min(), 2)
    resistance = round(recent["High"].max(), 2)
    return {"support": support, "resistance": resistance}


def get_signal(df: pd.DataFrame) -> dict:
    """综合买卖信号"""
    if len(df) < 26:
        return {"buy_signals": [], "sell_signals": [], "recommendation": "观望"}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    rsi = latest["RSI_14"]

    buy = []
    sell = []

    # MACD 信号
    macd_info = detect_macd_cross(df["MACD_hist"])
    if macd_info["status"] == "golden_cross":
        buy.append("MACD金叉")
    elif macd_info["status"] == "death_cross":
        sell.append("MACD死叉")

    # RSI 信号
    if rsi < 30:
        buy.append(f"RSI={rsi:.1f} 超卖")
    elif rsi > 70:
        sell.append(f"RSI={rsi:.1f} 超买")

    # 均线信号
    if latest["Close"] > latest["MA20"] and prev["Close"] <= prev["MA20"]:
        buy.append("股价站上MA20")
    elif latest["Close"] < latest["MA20"] and prev["Close"] >= prev["MA20"]:
        sell.append("股价跌破MA20")

    # 综合建议
    score = len(buy) - len(sell)
    if score >= 2:
        rec = "买入"
    elif score <= -2:
        rec = "卖出"
    else:
        rec = "观望"

    return {
        "buy_signals": buy,
        "sell_signals": sell,
        "recommendation": rec,
        "signal_score": score,
    }


# ─────────────────────────────────────────
# 接口 1：股票基本信息
# ─────────────────────────────────────────

@app.get("/stock/info", summary="获取股票基本信息")
def get_stock_info(
    symbol: str = Query(..., description="股票代码，如 AAPL / 0700.HK"),
    market: str = Query("auto", description="市场：us / hk / auto"),
):
    sym = normalize_symbol(symbol, market)
    try:
        info = safe_info(sym)
        hist = safe_history(sym, period="5d")

        if hist.empty:
            raise HTTPException(status_code=404, detail="未找到该股票数据")

        current_price = round(hist["Close"].iloc[-1], 2) if not hist.empty else None
        prev_close = round(hist["Close"].iloc[-2], 2) if len(hist) >= 2 else None
        change_pct = (
            round((current_price - prev_close) / prev_close * 100, 2)
            if prev_close else None
        )

        return {
            "symbol": sym,
            "name": info.get("longName") or info.get("shortName", "暂无"),
            "current_price": current_price,
            "change_percent": change_pct,
            "prev_close": prev_close,
            "open": round(hist["Open"].iloc[-1], 2) if not hist.empty else None,
            "day_high": round(hist["High"].iloc[-1], 2) if not hist.empty else None,
            "day_low": round(hist["Low"].iloc[-1], 2) if not hist.empty else None,
            "volume": int(hist["Volume"].iloc[-1]) if not hist.empty else None,
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据失败: {str(e)}")


# ─────────────────────────────────────────
# 接口 2：K线数据 + 技术指标
# ─────────────────────────────────────────

@app.get("/stock/kline", summary="获取K线数据和技术指标")
def get_kline(
    symbol: str = Query(..., description="股票代码"),
    period: str = Query("6mo", description="周期：1mo/3mo/6mo/1y/2y"),
    interval: str = Query("1d", description="K线间隔：1d/1wk/1mo"),
):
    sym = normalize_symbol(symbol, "auto")
    try:
        df = safe_download(sym, period=period, interval=interval)

        if df.empty:
            raise HTTPException(status_code=404, detail="未找到K线数据")

        # 计算技术指标
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["RSI_14"] = calc_rsi(df["Close"], 14)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = calc_macd(df["Close"])

        # 最近60条，转 JSON
        recent = df.tail(60).copy()
        recent.index = recent.index.strftime("%Y-%m-%d")
        recent = recent.replace({np.nan: None})

        result = []
        for date, row in recent.iterrows():
            result.append({
                "date": date,
                "open": round(row["Open"], 2) if row["Open"] else None,
                "high": round(row["High"], 2) if row["High"] else None,
                "low": round(row["Low"], 2) if row["Low"] else None,
                "close": round(row["Close"], 2) if row["Close"] else None,
                "volume": int(row["Volume"]) if row["Volume"] else None,
                "ma5": round(row["MA5"], 2) if row["MA5"] else None,
                "ma20": round(row["MA20"], 2) if row["MA20"] else None,
                "ma60": round(row["MA60"], 2) if row["MA60"] else None,
                "rsi_14": round(row["RSI_14"], 2) if row["RSI_14"] else None,
                "macd": round(row["MACD"], 4) if row["MACD"] else None,
                "macd_hist": round(row["MACD_hist"], 4) if row["MACD_hist"] else None,
            })

        return {"symbol": sym, "count": len(result), "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取K线失败: {str(e)}")


# ─────────────────────────────────────────
# 接口 3：买卖信号分析（核心）
# ─────────────────────────────────────────

@app.get("/stock/signal", summary="综合买卖信号分析")
def get_signal_analysis(
    symbol: str = Query(...),
    market: str = Query("auto"),
):
    sym = normalize_symbol(symbol, market)
    try:
        # 获取6个月日线
        df = safe_download(sym, period="6mo", interval="1d")
        if df.empty:
            raise HTTPException(status_code=404, detail="未找到数据")

        # 计算指标
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        df["RSI_14"] = calc_rsi(df["Close"], 14)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = calc_macd(df["Close"])

        # 信号分析
        signal = get_signal(df)
        latest = df.iloc[-1]
        rsi_val = latest["RSI_14"]

        # 支撑/压力
        sr = find_support_resistance(df)

        # 均线状态
        ma_status = []
        if not pd.isna(latest["MA5"]):
            ma_status.append(f"MA5: {latest['MA5']:.2f}")
        if not pd.isna(latest["MA20"]):
            ma_status.append(f"MA20: {latest['MA20']:.2f}")
        if not pd.isna(latest["MA60"]):
            ma_status.append(f"MA60: {latest['MA60']:.2f}")

        ma_trend = "站上MA20" if latest["Close"] > latest["MA20"] else "跌破MA20"

        return {
            "symbol": sym,
            "analysis_date": df.index[-1].strftime("%Y-%m-%d"),
            "current_price": round(latest["Close"], 2),
            "rsi_14": round(rsi_val, 2) if not pd.isna(rsi_val) else None,
            "rsi_status": (
                "超卖(<30)" if rsi_val < 30
                else "超买(>70)" if rsi_val > 70
                else "正常"
            ),
            "macd_status": detect_macd_cross(df["MACD_hist"])["desc"],
            "ma_trend": ma_trend,
            "ma_values": ma_status,
            "support": sr["support"],
            "resistance": sr["resistance"],
            "buy_signals": signal["buy_signals"],
            "sell_signals": signal["sell_signals"],
            "recommendation": signal["recommendation"],
            "signal_score": signal["signal_score"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"信号分析失败: {str(e)}")


# ─────────────────────────────────────────
# 接口 4：完整分析报告（Coze 一键调用）
# ─────────────────────────────────────────

@app.get("/stock/analyze", summary="完整分析报告（一键调用）")
def full_analysis(
    symbol: str = Query(...),
    market: str = Query("auto"),
):
    """Coze Agent 主要调用这个接口，一次拿到所有分析数据"""
    sym = normalize_symbol(symbol, market)

    try:
        info = safe_info(sym)
        df = safe_download(sym, period="6mo", interval="1d")

        if df.empty:
            raise HTTPException(status_code=404, detail="未找到数据")

        # ── 基本信息 ──
        hist = safe_history(sym, period="5d")
        current_price = round(hist["Close"].iloc[-1], 2) if not hist.empty else None
        prev_close = round(hist["Close"].iloc[-2], 2) if len(hist) >= 2 else None
        change_pct = (
            round((current_price - prev_close) / prev_close * 100, 2)
            if prev_close and current_price else None
        )

        # ── 技术指标 ──
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["RSI_14"] = calc_rsi(df["Close"], 14)
        df["MACD"], df["MACD_signal"], df["MACD_hist"] = calc_macd(df["Close"])

        latest = df.iloc[-1]
        rsi_val = latest["RSI_14"]
        signal = get_signal(df)
        sr = find_support_resistance(df)
        macd_info = detect_macd_cross(df["MACD_hist"])

        # ── K线数据（最近60天，给 Coze 画图用）──
        recent = df.tail(60).copy()
        kline = []
        for i, (date, row) in enumerate(recent.iterrows()):
            kline.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2) if row["Open"] else None,
                "high": round(row["High"], 2) if row["High"] else None,
                "low": round(row["Low"], 2) if row["Low"] else None,
                "close": round(row["Close"], 2) if row["Close"] else None,
                "volume": int(row["Volume"]) if row["Volume"] else None,
                "ma5": round(row["MA5"], 2) if not pd.isna(row["MA5"]) else None,
                "ma20": round(row["MA20"], 2) if not pd.isna(row["MA20"]) else None,
                "rsi_14": round(row["RSI_14"], 2) if not pd.isna(row["RSI_14"]) else None,
                "macd_hist": round(row["MACD_hist"], 4) if not pd.isna(row["MACD_hist"]) else None,
            })

        return {
            "stock": {
                "symbol": sym,
                "name": info.get("longName") or info.get("shortName", sym),
                "current_price": current_price,
                "change_percent": change_pct,
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "currency": info.get("currency", "USD"),
            },
            "indicators": {
                "rsi_14": round(rsi_val, 2) if not pd.isna(rsi_val) else None,
                "rsi_status": (
                    "超卖" if rsi_val and rsi_val < 30
                    else "超买" if rsi_val and rsi_val > 70
                    else "正常"
                ),
                "macd_status": macd_info["desc"],
                "ma_trend": "站上MA20" if latest["Close"] > latest["MA20"] else "跌破MA20",
                "ma20": round(latest["MA20"], 2) if not pd.isna(latest["MA20"]) else None,
                "support": sr["support"],
                "resistance": sr["resistance"],
            },
            "signals": {
                "buy": signal["buy_signals"],
                "sell": signal["sell_signals"],
                "recommendation": signal["recommendation"],
            },
            "kline_60d": kline,
            "disclaimer": "数据来自公开来源，仅供参考，不构成投资建议。股市有风险，投资需谨慎。",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


# ─────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("🚀 Stock Analysis API 启动中...")
    print("📊 接口文档： http://localhost:8000/docs")
    print("🔧 测试接口： http://localhost:8000/stock/analyze?symbol=AAPL")
    uvicorn.run(app, host="0.0.0.0", port=8000)
