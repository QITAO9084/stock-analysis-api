from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI(
    title="Stock Analysis API",
    description="股票买卖点分析API - V2增强版（含交叉检测、KDJ、成交量确认、趋势判断）",
    version="2.0.0"
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
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
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
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
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3mo")
        
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

    try:
        ticker = yf.Ticker(symbol)

        # 获取股票信息
        info = ticker.info

        # 获取历史数据
        data = ticker.history(period="6mo")

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

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        data = ticker.history(period="6mo")

        if data.empty:
            raise HTTPException(status_code=404, detail="未找到股票数据")

        signal_data = get_trading_signal(data, symbol)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
