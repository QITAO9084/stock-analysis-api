"""
discover_pool.py v2.0 — Phase 1 升级：4因子综合评分
动态发现强势美股，写入 stock_pool_dynamic.json
供 batch_analyze 的 pool=dynamic 模式使用

V2.0 升级：
  - 排序从纯涨幅 → 4因子综合评分（涨幅20% + RSI适中30% + ADX30% + 均线排列20%）
  - 数据周期 5d → 1mo（支持 ADX 计算）
  - 自建 ADX/均线指标，不依赖 main.py

运行方式：
  python discover_pool.py              # 全量刷新（≈60-90秒）
  python discover_pool.py --fast      # 快速模式（只读缓存，≈1秒）

定时：建议每天 09:00 北京时间运行一次
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

# 修复 Windows 控制台 UTF-8 编码问题（emoji 打印需要）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ---- 路径 ---------------------------------------------------------------
# 优先用 PORTFOLIO_DIR 环境变量（Railway Volume 挂载点）
_import_os = __import__("os")
_PORTFOLIO_DIR = _import_os.environ.get("PORTFOLIO_DIR") or _import_os.environ.get("PORTFOLIO_PATH")
if _PORTFOLIO_DIR:
    _BASE_DIR = Path(_PORTFOLIO_DIR)
else:
    _BASE_DIR = Path(__file__).parent

_CACHE_FILE = _BASE_DIR / "stock_pool_dynamic.json"

# ---- 100只股票池（与 us_stock_pool_100.md 同步） ----------------
# 按板块分类，便于后续做行业归因
SECTORS = {
    "信息技术": "AAPL,MSFT,NVDA,AMD,INTC,QCOM,CRM,ADBE,ORCL,CSCO,IBM,TXN,AVGO,PLTR,SNOW",
    "金融": "JPM,BAC,GS,MS,V,MA,BLK,SCHW,C,WFC,AXP,SPGI",
    "医疗保健": "JNJ,UNH,PFE,MRK,ABBV,TMO,ABT,LLY,BMY,AMGN,GILD,ISRG",
    "非必需消费": "TSLA,AMZN,HD,NKE,MCD,SBUX,TGT,LOW,EBAY,ETSY,RBLX,BKNG",
    "通信服务": "GOOGL,META,NFLX,T,TMUS,VZ,CMCSA,DIS,SPOT,EA",
    "工业": "BA,CAT,GE,HON,UPS,LMT,RTX,DE,EAD,FDX",
    "必需消费": "KO,PEP,PG,WMT,COST,PM,MO,CL",
    "能源": "XOM,CVX,COP,EOG,SLB,OXY,PSX,VLO",
    "公用事业": "NEE,DUK,SO,D,EXC,AEP",
    "房地产": "PLD,AMT,EQIX,SPG,O",
    "材料": "LIN,APD",
}
ALL_100 = []
for symbols in SECTORS.values():
    for s in symbols.split(","):
        s = s.strip()
        if s and s not in ALL_100:
            ALL_100.append(s)


def beijing_now():
    return datetime.now(timezone(timedelta(hours=8)))


# ============================================================================
# 技术指标计算（V2.0 自建，不依赖 main.py）
# ============================================================================

def _calc_rsi(prices: list, period: int = 14) -> int:
    """简易 RSI 计算，返回 0-100 整数"""
    if len(prices) < period + 1:
        return 50
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return int(rsi)


def _calc_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """
    计算 ADX（Average Directional Index），返回浮点数。
    ADX >= 40: 强趋势, >= 25: 中等趋势, < 20: 无趋势/盘整
    """
    n = len(highs)
    if n < period + 2:
        return 20.0  # 数据不足，返回中性默认值

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        h, l = highs[i], lows[i]
        c_prev = closes[i - 1]
        h_prev, l_prev = highs[i - 1], lows[i - 1]

        # True Range
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)

        # Directional Movement
        up_move = h - h_prev
        down_move = l_prev - l

        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        else:
            plus_dm = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm = down_move
        else:
            minus_dm = 0.0

        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    # Wilder's smoothing: SMA for first period, then EMA with alpha=1/period
    def _wilder_smooth(values, p):
        if len(values) < p:
            return [sum(values) / len(values)] if values else [0]
        smoothed = [sum(values[:p]) / p]
        for i in range(p, len(values)):
            smoothed.append((smoothed[-1] * (p - 1) + values[i]) / p)
        return smoothed

    atr_smooth = _wilder_smooth(tr_list, period)
    pdi_smooth = _wilder_smooth(plus_dm_list, period)
    mdi_smooth = _wilder_smooth(minus_dm_list, period)

    dx_list = []
    for i in range(len(atr_smooth)):
        if atr_smooth[i] == 0:
            dx_list.append(0.0)
            continue
        plus_di = 100 * pdi_smooth[i] / atr_smooth[i]
        minus_di = 100 * mdi_smooth[i] / atr_smooth[i]
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_list.append(0.0)
        else:
            dx_list.append(100 * abs(plus_di - minus_di) / di_sum)

    if not dx_list:
        return 20.0

    adx_smooth = _wilder_smooth(dx_list, period)
    return round(adx_smooth[-1], 1)


def _calc_ma_align(closes: list) -> str:
    """
    判断均线排列方向（MA5/MA10/MA20）。
    返回: "bullish" | "mild_bullish" | "neutral" | "mild_bearish" | "bearish"
    """
    n = len(closes)
    if n < 20:
        return "neutral"

    ma5 = sum(closes[-5:]) / 5
    ma10 = sum(closes[-10:]) / 10
    ma20 = sum(closes[-20:]) / 20

    if ma5 > ma10 > ma20:
        return "bullish"
    elif ma5 > ma10:
        return "mild_bullish"
    elif ma5 < ma10 < ma20:
        return "bearish"
    elif ma5 < ma10:
        return "mild_bearish"
    else:
        return "neutral"


def _composite_score(change_pct_5d: float, rsi: int, adx: float, ma_align: str) -> float:
    """
    4 因子综合评分（0-100），按权重加权：
    - 5日涨幅 momentum (20%)
    - RSI 适中度 (30%)
    - ADX 趋势强度 (30%)
    - 均线多头排列 (20%)
    """

    # 1. 涨幅得分（0-100）：涨幅越大越好，>10% 封顶
    momentum_score = min(max(change_pct_5d / 10 * 100, 0), 100)

    # 2. RSI 适中度得分（0-100）
    # 理想区间 40-60 = 100 分；两极递减
    if 40 <= rsi <= 60:
        rsi_score = 100.0
    elif 30 <= rsi < 40:
        rsi_score = 50 + (rsi - 30) / 10 * 50
    elif 60 < rsi <= 70:
        rsi_score = 50 + (70 - rsi) / 10 * 50
    elif 20 <= rsi < 30:
        rsi_score = 20 + (rsi - 20) / 10 * 30
    elif 70 < rsi <= 80:
        rsi_score = 20 + (80 - rsi) / 10 * 30
    else:
        # < 20 或 > 80：给低分但不为零（极端情况也有爆发力）
        rsi_score = max(rsi / 80 * 20, 0) if rsi < 20 else max((100 - rsi) / 20 * 20, 0)

    # 3. ADX 趋势强度得分（0-100）
    # >= 40 满分, >= 25 线性, < 20 低分
    if adx >= 40:
        adx_score = 100.0
    elif adx >= 25:
        adx_score = 60 + (adx - 25) / 15 * 40
    elif adx >= 20:
        adx_score = (adx - 20) / 5 * 60
    else:
        adx_score = max(adx / 20 * 30, 10)

    # 4. 均线排列得分（0-100）
    ma_map = {
        "bullish": 100,
        "mild_bullish": 70,
        "neutral": 40,
        "mild_bearish": 20,
        "bearish": 0,
    }
    ma_score = ma_map.get(ma_align, 40)

    composite = (
        momentum_score * 0.20 +
        rsi_score * 0.30 +
        adx_score * 0.30 +
        ma_score * 0.20
    )
    return round(max(composite, 5), 1)


def _calc_indicators(df) -> dict:
    """
    从单只股票的 DataFrame（含 OHLCV）计算所有指标。
    返回同 fetch_batch 的 dict 结构。
    """
    closes = [float(x) for x in df["Close"].tolist()]
    highs = [float(x) for x in df["High"].tolist()]
    lows = [float(x) for x in df["Low"].tolist()]

    n = len(closes)
    if n < 2:
        return None

    close = closes[-1]
    # 5 日涨幅：用最近 5 根 K 线
    close_5d_idx = max(0, n - 5)
    close_5d_ago = closes[close_5d_idx]
    change_pct_5d = (close - close_5d_ago) / close_5d_ago * 100 if close_5d_ago != 0 else 0

    rsi = _calc_rsi(closes, 14)
    adx = _calc_adx(highs, lows, closes, 14)
    ma_align = _calc_ma_align(closes)
    score = _composite_score(change_pct_5d, rsi, adx, ma_align)

    volumes = [float(x) for x in df["Volume"].dropna().tail(5).tolist()]
    vol_avg = sum(volumes) / len(volumes) if volumes else 0

    return {
        "close": round(close, 2),
        "change_pct_5d": round(change_pct_5d, 2),
        "volume_avg": int(vol_avg),
        "rsi": rsi,
        "adx": adx,
        "ma_align": ma_align,
        "score": score,
    }


# ============================================================================
# 数据获取（V2.0: period 5d → 1mo）
# ============================================================================

def fetch_batch(symbols: list, period: str = "1mo", max_retries: int = 2) -> dict:
    """
    批量下载 1mo 日线数据，自动分批（每批 20 只）避免超时。
    对每只股票计算 4 因子综合评分。
    返回 {symbol: {close, change_pct_5d, volume_avg, rsi, adx, ma_align, score}}，
    失败的符号不包含在结果中（不阻断整体流程）。
    """
    yf = __import__("yfinance")
    result = {}
    batch_size = 20
    total = len(symbols)

    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        tickers_str = " ".join(batch)
        for attempt in range(max_retries + 1):
            try:
                data = yf.download(
                    tickers_str,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    auto_adjust=True,
                )
                break
            except Exception as e:
                if attempt >= max_retries:
                    print(f"  ⚠️ 批次 {i // batch_size + 1} 失败：{e}")
                    data = None
                else:
                    time.sleep(2 * (attempt + 1))

        if data is None:
            continue

        # 解析每只股票的数据，计算指标
        for sym in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    if sym not in data.columns.levels[0]:
                        continue
                    df = data[sym]

                if df is None or len(df) < 2:
                    continue

                # V2.0: 调用统一指标计算，替代手动逐一计算
                indicators = _calc_indicators(df)
                if indicators:
                    result[sym] = indicators
            except Exception as e:
                print(f"  ⚠️ {sym} 解析/计算失败：{e}")
                continue

        # 限速：每次批次间隔 0.5 秒，避免 yfinance 限流
        time.sleep(0.5)

    return result


# ============================================================================
# 主逻辑
# ============================================================================

def run_discover(fast: bool = False) -> dict:
    """
    主函数：扫描 100 只股票，按 4 因子综合评分排序，返回 Top 30。
    fast=True：直接读缓存，不重新计算。
    """
    if fast and _CACHE_FILE.exists():
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"⚡ 快速模式：读取缓存（{cache.get('updated', '?')}）")
        return cache

    print(f"🔍 开始扫描 {len(ALL_100)} 只美股（V2.0 4因子综合评分）...")
    print(f"⏳ 预计耗时 60-90 秒（下载 1mo 日线 + 指标计算）")

    t0 = time.time()
    data = fetch_batch(ALL_100)
    elapsed = time.time() - t0

    print(f"✅ 下载+计算完成（{elapsed:.1f}秒），成功 {len(data)}/{len(ALL_100)} 只")

    if not data:
        print("❌ 没有获取到任何数据，返回空结果")
        return {"updated": beijing_now().strftime("%Y-%m-%d %H:%M"),
                "total_scanned": 0, "top_30": [], "sector_summary": {}}

    # V2.0: 按综合评分降序排列（原为按涨幅）
    sorted_items = sorted(data.items(), key=lambda x: x[1].get("score", 0), reverse=True)
    top_30 = []
    for rank, (sym, info) in enumerate(sorted_items[:30], 1):
        # 查板块
        sector = "其他"
        for s, symbols in SECTORS.items():
            if sym in symbols:
                sector = s
                break
        top_30.append({
            "rank": rank,
            "symbol": sym,
            "close": info["close"],
            "change_pct_5d": info["change_pct_5d"],
            "rsi": info["rsi"],
            "adx": info["adx"],
            "ma_align": info["ma_align"],
            "score": info["score"],
            "volume_avg": info["volume_avg"],
            "sector": sector,
        })

    # 行业归因：统计 Top 30 里各板块数量和平均评分
    sector_data = {}
    for item in top_30:
        s = item["sector"]
        if s not in sector_data:
            sector_data[s] = {"count": 0, "scores": []}
        sector_data[s]["count"] += 1
        sector_data[s]["scores"].append(item["score"])

    sector_summary = {}
    for s, d in sorted(sector_data.items(), key=lambda x: -x[1]["count"]):
        sector_summary[s] = {
            "count": d["count"],
            "avg_score": round(sum(d["scores"]) / len(d["scores"]), 1),
        }

    # 统计评分分布
    score_dist = {"A(>=70)": 0, "B(55-69)": 0, "C(40-54)": 0, "D(<40)": 0}
    for item in top_30:
        s = item["score"]
        if s >= 70:
            score_dist["A(>=70)"] += 1
        elif s >= 55:
            score_dist["B(55-69)"] += 1
        elif s >= 40:
            score_dist["C(40-54)"] += 1
        else:
            score_dist["D(<40)"] += 1

    result = {
        "version": "2.0",
        "algorithm": "4-factor composite (momentum 20% + RSI 30% + ADX 30% + MA align 20%)",
        "updated": beijing_now().strftime("%Y-%m-%d %H:%M"),
        "total_scanned": len(data),
        "score_distribution": score_dist,
        "top_30": top_30,
        "sector_summary": sector_summary,
    }

    # 写文件
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已写入 {_CACHE_FILE}（{len(top_30)} 只）")

    return result


def main():
    parser = argparse.ArgumentParser(description="动态发现强势美股 V2.0（4因子综合评分）")
    parser.add_argument("--fast", action="store_true", help="快速模式（只读缓存）")
    args = parser.parse_args()

    result = run_discover(fast=args.fast)

    # 终端输出摘要
    print(f"\n📊 评分分布：{result.get('score_distribution', {})}")
    print(f"\n📊 Top 10（按综合评分）：")
    for item in result["top_30"][:10]:
        print(f"  #{item['rank']:2d} {item['symbol']:5s}  "
              f"评分{item['score']:5.1f}  "
              f"涨幅{item['change_pct_5d']:+5.1f}%  "
              f"ADX={item['adx']:4.1f}  RSI={item['rsi']:2d}  "
              f"均线={item['ma_align']:12s}  "
              f"({item['sector']})")

    print(f"\n📋 行业分布（Top 30）：")
    for sector, info in result["sector_summary"].items():
        print(f"  {sector}：{info['count']} 只（均分 {info['avg_score']}）")


if __name__ == "__main__":
    main()
