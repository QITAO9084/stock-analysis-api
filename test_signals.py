"""V5.20.24 信号一致性验证 — 6只代表标的"""
import urllib.request
import json
import sys

STOCKS = [
    ("TSLA", "us", "ADX<25震荡"),
    ("GOOG", "us", "ADX>=25强趋势"),
    ("NVDA", "us", "AI龙头"),
    ("AAPL", "us", "消费电子"),
    ("0700.HK", "hk", "港股代表"),
    ("600519.SS", "cn", "A股代表"),
]

BASE = "https://web-production-e9fcc.up.railway.app"

def check(symbol, market, label):
    url = f"{BASE}/stock/analyze2?symbol={symbol}&market={market}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "test/5.20.24"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[{symbol}] ERROR: {e}")
        return {"status": "ERROR", "symbol": symbol}

    sig = data.get("signal", "?")
    tp = data.get("trade_point", "?")
    tpc = data.get("trade_point_cn", "?")
    adx = data.get("adx", "?")
    adx_t = data.get("adx_trend", "?")
    conf = data.get("confidence", "?")
    trend = data.get("trend_direction", "?")

    problems = []

    # 规则1: signal和trade_point方向必须一致
    if sig == "BUY" and tp == "hold":
        problems.append("BUY vs hold")
    if sig == "BUY" and tp == "sell":
        problems.append("BUY vs sell")
    if sig == "SELL" and tp == "buy":
        problems.append("SELL vs buy")
    if sig == "SELL" and tp == "hold":
        problems.append("SELL vs hold")
    if sig == "BUY" and tp == "strong_sell":
        problems.append("BUY vs strong_sell")
    if sig == "SELL" and tp == "strong_buy":
        problems.append("SELL vs strong_buy")

    # 规则2: ADX<25时必须为NEUTRAL/hold
    if isinstance(adx, (int, float)) and adx < 25:
        if sig not in ("NEUTRAL", "HOLD"):
            problems.append(f"ADX<25但sig={sig}")

    # 规则3: NEUTRAL信号必须是LOW置信度
    if sig == "NEUTRAL" and conf != "LOW":
        problems.append(f"NEUTRAL但conf={conf}")

    status = "PASS" if not problems else "FAIL:" + ",".join(problems)

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
        "trend": trend,
        "problems": problems,
    }
    print(f"[{status}] {symbol:>10s} ({label:15s}) | ADX={adx:>6} {adx_t:12s} | sig={sig:>7s} tp={tp:>10s} conf={conf:>6s}")
    return result

def main():
    print("=" * 120)
    print("V5.20.24 信号一致性验证")
    print("=" * 120)

    results = []
    for symbol, market, label in STOCKS:
        r = check(symbol, market, label)
        results.append(r)

    print("=" * 120)
    failures = [r for r in results if not r["status"].startswith("PASS")]
    errors = [r for r in results if r["status"] == "ERROR"]
    passes = [r for r in results if r["status"].startswith("PASS")]

    print(f"\nSummary: {len(passes)} PASS, {len(failures)} FAIL, {len(errors)} ERROR")

    if failures:
        print("\n!!! FAILURES:")
        for r in failures:
            print(f"  - {r['symbol']}: {r['status']}")

    return 0 if not failures else 1

if __name__ == "__main__":
    sys.exit(main())
