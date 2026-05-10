from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI(
    title="Stock Analysis API",
    description="股票买卖点分析API",
    version="1.0.0"
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
    """计算RSI指标"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not pd.isna(rs.iloc[-1]) else 50

def calculate_macd(data):
    """计算MACD指标"""
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return {
        "macd": round(macd.iloc[-1], 4),
        "signal": round(signal.iloc[-1], 4),
        "histogram": round(histogram.iloc[-1], 4)
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

def get_trading_signal(data, symbol):
    """生成交易信号（修复版：买卖信号不矛盾）"""
    current_price = data['Close'].iloc[-1]
    ma5 = data['Close'].rolling(window=5).mean().iloc[-1]
    ma10 = data['Close'].rolling(window=10).mean().iloc[-1]
    ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
    
    rsi = calculate_rsi(data)
    macd_data = calculate_macd(data)
    kdj_data = calculate_kdj(data)
    boll = calculate_bollinger_bands(data)
    
    buy_score = 0
    sell_score = 0
    buy_signals = []
    sell_signals = []
    
    # 买入信号
    if rsi < 30:
        buy_signals.append("RSI超卖（<30），可能反弹")
        buy_score += 2
    elif rsi < 45:
        buy_signals.append("RSI偏低，可考虑建仓")
        buy_score += 1
    
    if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
        buy_signals.append("MACD金叉，趋势偏多")
        buy_score += 2
    
    if current_price < boll['lower']:
        buy_signals.append("价格触及布林带下轨，超卖")
        buy_score += 1
    
    if current_price > ma5 and ma5 > ma10 and ma10 > ma20:
        buy_signals.append("均线多头排列，趋势向上")
        buy_score += 2
    
    # 卖出信号
    if rsi > 70:
        sell_signals.append("RSI超买（>70），注意风险")
        sell_score += 2
    elif rsi > 60:
        sell_signals.append("RSI偏高，可考虑减仓")
        sell_score += 1
    
    if macd_data['macd'] < macd_data['signal'] and macd_data['histogram'] < 0:
        sell_signals.append("MACD死叉，趋势偏空")
        sell_score += 2
    
    if current_price > boll['upper']:
        sell_signals.append("价格触及布林带上轨，超买")
        sell_score += 1
    
    if current_price < ma5 and ma5 < ma10 and ma10 < ma20:
        sell_signals.append("均线空头排列，趋势向下")
        sell_score += 2
    
    # 根据评分决定最终信号（只输出获胜方的信号）
    if buy_score > sell_score and buy_score >= 2:
        signal_type = "BUY"
        signals = buy_signals
        confidence = "HIGH" if buy_score >= 4 else "MEDIUM"
    elif sell_score > buy_score and sell_score >= 2:
        signal_type = "SELL"
        signals = sell_signals
        confidence = "HIGH" if sell_score >= 4 else "MEDIUM"
    else:
        signal_type = "HOLD"
        confidence = "LOW"
        signals = []
        if len(buy_signals) == 0 and len(sell_signals) == 0:
            signals.append("无明显买卖信号，建议观望")
        else:
            signals = buy_signals + sell_signals
    
    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "signal": signal_type,
        "confidence": confidence,
        "signals": signals,
        "indicators": {
            "rsi": round(rsi, 2),
            "macd": macd_data,
            "kdj": kdj_data,
            "bollinger_bands": boll,
            "ma5": round(ma5, 2),
            "ma10": round(ma10, 2),
            "ma20": round(ma20, 2)
        },
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
