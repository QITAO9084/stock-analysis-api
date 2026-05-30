"""V5.20.25 信号一致性验证 — 10只标的 + HOLD→NEUTRAL规则 + RSI/KDJ/K线覆盖"""
import urllib.request
import json
import sys

STOCKS = [
    # === 美股 (6只) ===
    ("TSLA", "us", "ADX<25震荡"),
    ("GOOG", "us", "ADX>=25强趋势"),
    ("NVDA", "us", "AI龙头"),
    ("AAPL", "us", "消费电子"),
    ("AMD", "us", "半导体"),       # 新增
    ("JPM", "us", "金融代表"),     # 新增
    # === 港股 (1只) ===
    ("0700.HK", "hk", "港股代表"),
    # === A股 (2只) ===
    ("600519.SS", "cn", "A股茅台"),
    ("300750.SZ", "cn", "A股宁德"),    # 新增：创业板
    # === 加密货币 (2只) ===
    ("BTC-USD", "crypto", "比特币"),   # 新增
    ("ETH-USD", "crypto", "以太坊"),   # 新增
]

BASE = "https://web-production-e9fcc.up.railway.app"

def check(symbol, market, label):
    url = f"{BASE}/stock/analyze2?symbol={symbol}&market={market}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "test/5.20.25"})
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[{symbol:>12s}] ERROR: {e}")
        return {"status": "ERROR", "symbol": symbol}

    sig = data.get("signal", "?")
    tp = data.get("trade_point", "?")
    tpc = data.get("trade_point_cn", "?")
    adx = data.get("adx", "?")
    adx_t = data.get("adx_trend", "?")
    conf = data.get("confidence", "?")
    rsi = data.get("rsi", "?")
    kdj_j = data.get("kdj_j", "?")
    pattern = data.get("pattern", "?")

    problems = []

    # 规则1: signal和trade_point方向必须一致
    contradictions = [
        ("BUY", "hold"), ("BUY", "sell"), ("BUY", "strong_sell"),
        ("SELL", "buy"), ("SELL", "hold"), ("SELL", "strong_buy"),
    ]
    for s, t in contradictions:
        if sig == s and tp == t:
            problems.append(f"{sig} vs {tp}")

    # 规则2: ADX<25时必须为NEUTRAL/hold
    if isinstance(adx, (int, float)) and adx < 25:
        if sig not in ("NEUTRAL",):
            problems.append(f"ADX<25但sig={sig}")

    # 规则3: NEUTRAL信号必须是LOW置信度
    if sig == "NEUTRAL" and conf != "LOW":
        problems.append(f"NEUTRAL但conf={conf}")

    # 规则4 (V5.20.25新增): 禁止 HOLD 泄露，只允许 BUY/SELL/NEUTRAL
    if sig == "HOLD":
        problems.append("sig=HOLD应改为NEUTRAL")

    # 规则5: NEUTRAL + hold 必须配套
    if sig == "NEUTRAL" and tp != "hold":
        problems.append(f"NEUTRAL但tp={tp}")

    status = "PASS" if not problems else "FAIL:" + ",".join(problems)

    # 构建辅助指标信息
    extra = []
    if rsi != "?":
        extra.append(f"RSI={rsi}")
    if kdj_j != "?":
        extra.append(f"J={kdj_j}")
    if pattern != "?" and pattern:
        extra.append(f"形态={pattern}")
    extra_str = " | ".join(extra) if extra else ""

    result = {
        "status": status,
        "symbol": symbol,
        "label": label,
        "adx": adx,
        "adx_trend": adx_t,
        "signal": sig,
        "trade_point": tp,
        "trade_point_cn": tpc,
        "confidence": conf,
        "rsi": rsi,
        "kdj_j": kdj_j,
        "pattern": pattern,
        "problems": problems,
        "extra": extra_str,
    }
    print(f"[{status:>4s}] {symbol:>12s} ({label:12s}) | ADX={adx:>6} {adx_t:12s} | sig={sig:>7s} tp={tp:>10s} conf={conf:>6s} | {extra_str}")
    return result

def main():
    print("=" * 120)
    print("V5.20.25 信号一致性验证（HOLD→NEUTRAL标准化 + 10标的扩展矩阵）")
    print("=" * 120)

    results = []
    for symbol, market, label in STOCKS:
        r = check(symbol, market, label)
        results.append(r)

    print("=" * 120)
    failures = [r for r in results if not r["status"].startswith("PASS")]
    errors = [r for r in results if r["status"] == "ERROR"]
    passes = [r for r in results if r["status"].startswith("PASS")]

    print(f"\n{'='*60}")
    print(f"Summary: {len(passes)} PASS, {len(failures)} FAIL, {len(errors)} ERROR  (共 {len(results)} 只)")
    print(f"{'='*60}")

    # 按市场分类统计
    markets = {}
    for r in results:
        m = r.get("symbol", "?").split(".")[-1] if "." in r.get("symbol", "") else r.get("symbol", "?")
        if r["status"] == "ERROR":
            m = "ERROR"
        markets[m] = markets.get(m, 0) + 1

    # RSI 极值汇总
    rsi_extremes = [r for r in results if isinstance(r.get("rsi"), (int, float)) and (r["rsi"] > 70 or r["rsi"] < 30)]
    if rsi_extremes:
        print(f"\nRSI 极值标的（超买>70 / 超卖<30）:")
        for r in rsi_extremes:
            print(f"  {r['symbol']:>12s}: RSI={r['rsi']}  sig={r['signal']}  tp={r['trade_point']}")

    # 形态触发标的
    patterns = [r for r in results if r.get("pattern") and r["pattern"] != "?"]
    if patterns:
        print(f"\nK线形态触发标的:")
        for r in patterns:
            print(f"  {r['symbol']:>12s}: {r['pattern']}  sig={r['signal']}")

    if failures:
        print(f"\n!!! FAILURES ({len(failures)}):")
        for r in failures:
            print(f"  - {r['symbol']}: {r['status']}")

    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
