"""
discover_pool.py v1.0
动态发现强势美股，写入 stock_pool_dynamic.json
供 batch_analyze 的 pool=dynamic 模式使用

运行方式：
  python discover_pool.py              # 全量刷新（≈60秒）
  python discover_pool.py --fast      # 快速模式（只读缓存，≈1秒）

定时：建议每天 09:00 北京时间运行一次
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

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


def fetch_batch(symbols: list, period: str = "5d", max_retries: int = 2) -> dict:
    """
    批量下载，自动分批（每批20只）避免超时。
    返回 {symbol: {close, change_pct, volume_avg, rsi}}，
    失败的符号返回 None（不阻断整体流程）。
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

        # 解析 data（多股票返回 MultiIndex DataFrame）
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

                close = float(df["Close"].iloc[-1])
                open_5d = float(df["Close"].iloc[0])
                change_pct = (close - open_5d) / open_5d * 100

                volumes = df["Volume"].dropna().tail(5).tolist()
                vol_avg = sum(volumes) / len(volumes) if volumes else 0

                # 简易 RSI(14)
                rsi = _calc_rsi(df["Close"].tail(20).tolist())

                result[sym] = {
                    "close": round(close, 2),
                    "change_pct_5d": round(change_pct, 2),
                    "volume_avg": int(vol_avg),
                    "rsi": rsi,
                }
            except Exception as e:
                print(f"  ⚠️ {sym} 解析失败：{e}")
                continue

        # 限速：每次批次间隔 0.5 秒，避免 yfinance 限流
        time.sleep(0.5)

    return result


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


def run_discover(fast: bool = False) -> dict:
    """
    主函数：扫描 100 只股票，返回 Top 30 强势股。
    fast=True：直接读缓存，不重新计算。
    """
    if fast and _CACHE_FILE.exists():
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"⚡ 快速模式：读取缓存（{cache.get('updated', '?')}）")
        return cache

    print(f"🔍 开始扫描 {len(ALL_100)} 只美股...")
    print(f"⏳ 预计耗时 40-70 秒（分批下载）")

    t0 = time.time()
    data = fetch_batch(ALL_100, period="5d")
    elapsed = time.time() - t0

    print(f"✅ 下载完成（{elapsed:.1f}秒），成功 {len(data)}/{len(ALL_100)} 只")

    if not data:
        print("❌ 没有获取到任何数据，返回空结果")
        return {"updated": beijing_now().strftime("%Y-%m-%d %H:%M"),
                "top_30": [], "sector_summary": {}}

    # 按 5 日涨幅排序，取 Top 30
    sorted_items = sorted(data.items(), key=lambda x: x[1]["change_pct_5d"], reverse=True)
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
            "volume_avg": info["volume_avg"],
            "sector": sector,
        })

    # 行业归因：统计 Top 30 里各板块数量
    sector_cnt = {}
    for item in top_30:
        s = item["sector"]
        sector_cnt[s] = sector_cnt.get(s, 0) + 1

    result = {
        "updated": beijing_now().strftime("%Y-%m-%d %H:%M"),
        "total_scanned": len(data),
        "top_30": top_30,
        "sector_summary": dict(sorted(sector_cnt.items(), key=lambda x: -x[1])),
    }

    # 写文件
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已写入 {_CACHE_FILE}（{len(top_30)} 只）")

    return result


def main():
    parser = argparse.ArgumentParser(description="动态发现强势美股")
    parser.add_argument("--fast", action="store_true", help="快速模式（只读缓存）")
    args = parser.parse_args()

    result = run_discover(fast=args.fast)

    # 终端输出摘要
    print("\n📊 Top 10 强势股：")
    for item in result["top_30"][:10]:
        print(f"  #{item['rank']:2d} {item['symbol']:5s} "
              f"{item['change_pct_5d']:+5.1f}%  RSI={item['rsi']:2d}  "
              f"({item['sector']})")

    print(f"\n📋 行业分布（Top 30）：")
    for sector, cnt in result["sector_summary"].items():
        print(f"  {sector}：{cnt} 只")


if __name__ == "__main__":
    main()
