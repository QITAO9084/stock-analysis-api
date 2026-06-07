"""
discover_pool.py v2.1 — Phase 2.1 升级：动态因子权重
动态发现强势美股，写入 stock_pool_dynamic.json
供 batch_analyze 的 pool=dynamic 模式使用

V2.1 升级：
  - 新增市场状态检测（_detect_market_regime，基于 SPY）
  - 4因子权重随市场状态动态切换：
     牛市：涨跌幅35% RSI15% ADX35% 均线15%（降RSI惩罚，抓强势股）
     震荡：涨跌幅20% RSI30% ADX30% 均线20%（稳定权重）
     熊市：涨跌幅15% RSI40% ADX25% 均线20%（抓超卖反弹）
  - 解决 V2.0 在牛市中 RSI 惩罚过重导致漏掉强势股的问题

V2.0 升级：
  - 排序从纯涨幅 → 4因子综合评分
  - 数据周期 5d → 1mo（支持 ADX 计算）
  - 自建 ADX/均线指标，不依赖 main.py

运行方式：
  python discover_pool.py                # 全量刷新（≈60-90秒）
  python discover_pool.py --fast        # 快速模式（只读缓存，≈1秒）
  python discover_pool.py --show-regime # 仅显示当前市场状态

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

# ---- 市场基准（用于检测市场状态）-----------------------------------------
MARKET_BENCHMARK = "SPY"

# ---- 动态权重矩阵 ---------------------------------------------------------
# 根据市场状态（regime）切换 4 因子权重
WEIGHT_MATRIX = {
    "bull":   {"momentum": 0.35, "rsi": 0.15, "adx": 0.35, "ma": 0.15},
    "neutral":{"momentum": 0.20, "rsi": 0.30, "adx": 0.30, "ma": 0.20},
    "bear":   {"momentum": 0.15, "rsi": 0.40, "adx": 0.25, "ma": 0.20},
}

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
# 市场状态检测（V2.1 新增）
# ============================================================================

def _detect_market_regime(df_spy) -> str:
    """
    根据 SPY 的均线、ADX、波动率判断当前市场状态。
    返回："bull" | "neutral" | "bear"

    判断逻辑（三层投票）：
      1. 均线投票：MA20 > MA60 且 MA20 斜率为正 → bull_vote=1；反之下跌 → bear_vote=-1
         加最小差异阈值 0.5%，避免微小波动误判
      2. ADX 投票：ADX >= 25 → 有趋势，跟随均线方向；ADX < 20 → 震荡，vote=0
      3. 波动率过滤：10日波幅 > 12% 且连续5日价格低于 MA20 → 降级为 bear
    综合：bull_vote >= 2 → bull；bear_vote <= -2 → bear；否则 neutral
    """
    closes = [float(x) for x in df_spy["Close"].tolist()]
    highs = [float(x) for x in df_spy["High"].tolist()]
    lows = [float(x) for x in df_spy["Low"].tolist()]
    n = len(closes)
    if n < 30:
        return "neutral"  # 数据不足，默认震荡

    # 1. 均线投票（加最小差异阈值）
    ma20 = sum(closes[-20:]) / 20
    ma60_list = closes[-60:] if n >= 60 else closes
    ma60 = sum(ma60_list) / len(ma60_list)
    # MA20 斜率（最近5天 vs 前5天）
    ma20_slope_positive = (sum(closes[-5:]) / 5 - sum(closes[-10:-5]) / 5) > 0

    # 最小差异阈值：MA 差异 < 0.5% 视为无趋势
    MIN_MA_DIFF_PCT = 0.005
    ma_diff_pct = abs(ma20 - ma60) / ma60 if ma60 != 0 else 0

    ma_vote = 0
    if ma_diff_pct >= MIN_MA_DIFF_PCT:
        if ma20 > ma60 and ma20_slope_positive:
            ma_vote = 1
        elif ma20 < ma60 and not ma20_slope_positive:
            ma_vote = -1

    # 2. ADX 投票（趋势强度）
    adx = _calc_adx(highs, lows, closes, 14)
    adx_strong = adx >= 25
    adx_weak = adx < 20

    # 3. 价格 vs MA20 位置（辅助判断）
    price_above_ma20 = closes[-1] > ma20
    # 连续5日价格低于 MA20（确认下跌趋势）
    price_below_ma20_sustained = False
    if n >= 25:
        ma20_list = [sum(closes[i-19:i+1]) / 20 for i in range(-5, 0) if i + 19 >= -n]
        if len(ma20_list) == 5:
            price_below_ma20_sustained = all(closes[-5 + i] < ma20_list[i] for i in range(5))

    # 综合判断
    score = 0
    if ma_vote == 1:
        score += 1
    elif ma_vote == -1:
        score -= 1

    if adx_strong:
        # 有强趋势，跟随价格方向
        if price_above_ma20:
            score += 1
        else:
            score -= 1
    elif not adx_weak:
        # ADX 中等，跟随 MA 方向
        if ma_vote!= 0:
            score += ma_vote

    # 波动率过滤（高波动可能意味着熊市或顶部）
    recent_range = (max(highs[-10:]) - min(lows[-10:])) / closes[-1] if closes[-1] != 0 else 0
    high_volatility = recent_range > 0.12  # 10日波幅 > 12%（原8%过于敏感）

    if score >= 2:
        regime = "bull"
    elif score <= -2:
        regime = "bear"
    else:
        regime = "neutral"

    # 高波动 + 连续5日价格低于 MA20 → 降级为 bear
    if regime == "neutral" and high_volatility and price_below_ma20_sustained:
        regime = "bear"

    return regime


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
        return 20.0

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, n):
        h, l = highs[i], lows[i]
        c_prev = closes[i - 1]
        h_prev, l_prev = highs[i - 1], lows[i - 1]

        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)

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


def _composite_score(
    change_pct_5d: float, rsi: int, adx: float,
    ma_align: str, regime: str = "neutral"
) -> float:
    """
    4 因子综合评分（0-100），权重根据市场状态动态切换。
    regime: "bull" | "neutral" | "bear"
    """
    weights = WEIGHT_MATRIX.get(regime, WEIGHT_MATRIX["neutral"])

    # 1. 涨幅得分（0-100）：涨幅越大越好，>15% 封顶（牛市宽容度更高）
    cap = 15.0 if regime == "bull" else 10.0
    momentum_score = min(max(change_pct_5d / cap * 100, 0), 100)

    # 2. RSI 适中度得分（0-100）
    # 牛市：理想区间扩大到 40-70（不轻易惩罚强势股）
    # 熊市：理想区间缩小到 30-50（抓超卖反弹）
    # 震荡：理想区间 40-60
    if regime == "bull":
        ideal_low, ideal_high = 40, 70
        penalty_low, penalty_high = 30, 80
    elif regime == "bear":
        ideal_low, ideal_high = 30, 50
        penalty_low, penalty_high = 20, 60
    else:  # neutral
        ideal_low, ideal_high = 40, 60
        penalty_low, penalty_high = 30, 70

    if ideal_low <= rsi <= ideal_high:
        rsi_score = 100.0
    elif penalty_low <= rsi < ideal_low:
        rsi_score = 50 + (rsi - penalty_low) / (ideal_low - penalty_low) * 50
    elif ideal_high < rsi <= penalty_high:
        rsi_score = 50 + (penalty_high - rsi) / (penalty_high - ideal_high) * 50
    elif 20 <= rsi < penalty_low:
        rsi_score = 20 + (rsi - 20) / (penalty_low - 20) * 30
    elif penalty_high < rsi <= 80:
        rsi_score = 20 + (80 - rsi) / (80 - penalty_high) * 30
    else:
        rsi_score = max(rsi / 80 * 20, 0) if rsi < 20 else max((100 - rsi) / 20 * 20, 0)

    # 3. ADX 趋势强度得分（0-100）
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
        momentum_score * weights["momentum"] +
        rsi_score * weights["rsi"] +
        adx_score * weights["adx"] +
        ma_score * weights["ma"]
    )
    return round(max(composite, 5), 1)


def _calc_indicators(df, regime: str = "neutral") -> dict:
    """
    从单只股票的 DataFrame（含 OHLCV）计算所有指标。
    返回同 fetch_batch 的 dict 结构。
    regime: 市场状态，用于动态评分
    """
    closes = [float(x) for x in df["Close"].tolist()]
    highs = [float(x) for x in df["High"].tolist()]
    lows = [float(x) for x in df["Low"].tolist()]

    n = len(closes)
    if n < 2:
        return None

    close = closes[-1]
    close_5d_idx = max(0, n - 5)
    close_5d_ago = closes[close_5d_idx]
    change_pct_5d = (close - close_5d_ago) / close_5d_ago * 100 if close_5d_ago != 0 else 0

    rsi = _calc_rsi(closes, 14)
    adx = _calc_adx(highs, lows, closes, 14)
    ma_align = _calc_ma_align(closes)
    score = _composite_score(change_pct_5d, rsi, adx, ma_align, regime)

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
# 数据获取（V2.1: 新增 SPY 数据获取用于 regime 检测）
# ============================================================================

def fetch_batch(symbols: list, period: str = "1mo", max_retries: int = 2,
                regime: str = "neutral") -> dict:
    """
    批量下载 1mo 日线数据，自动分批（每批 20 只）避免超时。
    对每只股票计算 4 因子综合评分（动态权重）。
    regime: 市场状态，传递给 _calc_indicators。
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

                # V2.1: 传入 regime 参数
                indicators = _calc_indicators(df, regime)
                if indicators:
                    result[sym] = indicators
            except Exception as e:
                print(f"  ⚠️ {sym} 解析/计算失败：{e}")
                continue

        # 限速：每次批次间隔 0.5 秒，避免 yfinance 限流
        time.sleep(0.5)

    return result


def fetch_spy_data(period: str = "3mo") -> object:
    """
    下载 SPY 数据用于市场状态检测。
    返回 DataFrame 或 None。
    """
    yf = __import__("yfinance")
    try:
        data = yf.download(
            MARKET_BENCHMARK,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if data is not None and len(data) >= 20:
            return data
    except Exception as e:
        print(f"  ⚠️ SPY 数据下载失败：{e}")
    return None


# ============================================================================
# 主逻辑
# ============================================================================

def run_discover(fast: bool = False, show_regime_only: bool = False) -> dict:
    """
    主函数：扫描 100 只股票，按 4 因子综合评分（动态权重）排序，返回 Top 30。
    fast=True：直接读缓存，不重新计算。
    show_regime_only=True：仅检测并显示市场状态，不扫描股票。
    """
    # 步骤 0：检测市场状态（V2.1 新增）
    print("📊 检测市场状态（基于 SPY）...")
    spy_df = fetch_spy_data()
    if spy_df is not None:
        regime = _detect_market_regime(spy_df)
        weights = WEIGHT_MATRIX[regime]
        regime_cn = {"bull": "🐂 牛市", "neutral": "📊 震荡", "bear": "🐻 熊市"}[regime]
        print(f"  市场状态：{regime_cn}（regime={regime}）")
        print(f"  动态权重：涨跌幅{weights['momentum']*100:.0f}%  "
              f"RSI{weights['rsi']*100:.0f}%  "
              f"ADX{weights['adx']*100:.0f}%  "
              f"均线{weights['ma']*100:.0f}%")
    else:
        regime = "neutral"
        print("  ⚠️ SPY 数据获取失败，使用默认权重（震荡）")

    if show_regime_only:
        return {"regime": regime, "weights": WEIGHT_MATRIX[regime]}

    if fast and _CACHE_FILE.exists():
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"⚡ 快速模式：读取缓存（{cache.get('updated', '?')}）")
        return cache

    print(f"🔍 开始扫描 {len(ALL_100)} 只美股（V2.1 动态权重）...")
    print(f"⏳ 预计耗时 60-90 秒（下载 1mo 日线 + 指标计算）")

    t0 = time.time()
    data = fetch_batch(ALL_100, regime=regime)
    elapsed = time.time() - t0

    print(f"✅ 下载+计算完成（{elapsed:.1f}秒），成功 {len(data)}/{len(ALL_100)} 只")

    if not data:
        print("❌ 没有获取到任何数据，返回空结果")
        return {"updated": beijing_now().strftime("%Y-%m-%d %H:%M"),
                "total_scanned": 0, "top_30": [], "sector_summary": {}}

    # V2.1: 按综合评分降序排列（动态权重）
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

    regime_cn_map = {"bull": "🐂 牛市", "neutral": "📊 震荡", "bear": "🐻 熊市"}
    result = {
        "version": "2.1",
        "algorithm": f"4-factor dynamic weights (regime={regime})",
        "regime": regime,
        "regime_cn": regime_cn_map.get(regime, regime),
        "weights": WEIGHT_MATRIX[regime],
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
    parser = argparse.ArgumentParser(description="动态发现强势美股 V2.1（动态因子权重）")
    parser.add_argument("--fast", action="store_true", help="快速模式（只读缓存）")
    parser.add_argument("--show-regime", action="store_true",
                        help="仅显示当前市场状态（不扫描股票）")
    args = parser.parse_args()

    result = run_discover(fast=args.fast, show_regime_only=args.show_regime)

    if args.show_regime:
        regime = result.get("regime", "neutral")
        weights = result.get("weights", WEIGHT_MATRIX["neutral"])
        regime_cn = {"bull": "🐂 牛市", "neutral": "📊 震荡", "bear": "🐻 熊市"}.get(regime, regime)
        print(f"\n📊 市场状态检测结果：")
        print(f"  regime = {regime}  ({regime_cn})")
        print(f"  动态权重：")
        print(f"    涨跌幅  {weights['momentum']*100:.0f}%")
        print(f"    RSI     {weights['rsi']*100:.0f}%")
        print(f"    ADX     {weights['adx']*100:.0f}%")
        print(f"    均线    {weights['ma']*100:.0f}%")
        return

    # 终端输出摘要
    regime = result.get("regime", "neutral")
    regime_cn = result.get("regime_cn", regime)
    print(f"\n📊 市场状态：{regime_cn}  |  算法版本：{result.get('version', '2.1')}")
    print(f"📊 评分分布：{result.get('score_distribution', {})}")
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
