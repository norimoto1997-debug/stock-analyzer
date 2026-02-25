"""
株式自動分析 → LINE通知システム
毎朝自動実行して候補銘柄をLINEに送る
"""

import os
import requests
import yfinance as yf
import pandas as pd
from anthropic import Anthropic
from datetime import datetime
import time

# ==========================================
# 設定エリア（ここだけ変更すればOK）
# ==========================================

LINE_TOKEN = os.environ.get("LINE_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

WATCHLIST = {
    "国内株": [
        "7203.T","6758.T","9984.T","6861.T","4063.T",
        "8035.T","6367.T","9432.T","8306.T","4519.T",
    ],
    "米株": [
        "AAPL","MSFT","NVDA","GOOGL","META",
        "AMZN","TSLA","AMD","SMCI","PLTR",
    ]
}

SCREENING_RULES = {
    "min_volume_ratio": 1.5,
    "min_price_change": 2.0,
    "rsi_min": 45,
    "rsi_max": 75,
}

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        if hist.empty or len(hist) < 25:
            return None
        close = hist["Close"]
        volume = hist["Volume"]
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2]
        price_change_pct = (current_price - prev_price) / prev_price * 100
        avg_volume_20 = volume.iloc[-21:-1].mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
        rsi = calc_rsi(close).iloc[-1]
        ema20 = close.ewm(span=20).mean().iloc[-1]
        ema60 = close.ewm(span=60).mean().iloc[-1]
        return {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "current_price": round(current_price, 2),
            "price_change_pct": round(price_change_pct, 2),
            "volume_ratio": round(volume_ratio, 2),
            "rsi": round(rsi, 1),
            "ema20": round(ema20, 2),
            "ema60": round(ema60, 2),
            "above_ema20": current_price > ema20,
            "above_ema60": current_price > ema60,
            "market_cap": info.get("marketCap", 0),
            "sector": info.get("sector", "不明"),
        }
    except Exception as e:
        print(f"  ⚠️ {ticker} データ取得失敗: {e}")
        return None

def screen_stocks(watchlist):
    candidates = []
    rules = SCREENING_RULES
    all_tickers = watchlist["国内株"] + watchlist["米株"]
    print(f"📡 {len(all_tickers)}銘柄をスクリーニング中...")
    for ticker in all_tickers:
        data = get_stock_data(ticker)
        if not data:
            continue
        checks = {
            "出来高急増": data["volume_ratio"] >= rules["min_volume_ratio"],
            "価格上昇": data["price_change_pct"] >= rules["min_price_change"],
            "RSI適正": rules["rsi_min"] <= data["rsi"] <= rules["rsi_max"],
            "トレンド": data["above_ema20"] and data["above_ema60"],
        }
        passed = sum(checks.values())
        if passed >= 3:
            data["checks"] = checks
            data["score"] = passed
            candidates.append(data)
        time.sleep(0.3)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates

def analyze_with_claude(candidates):
    if not candidates:
        return "本日の候補銘柄はありませんでした。"
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    stock_info = ""
    for s in candidates[:5]:
        checks_str = " / ".join([k for k, v in s["checks"].items() if v])
        stock_info += f"""
■ {s['name']} ({s['ticker']})
  現在値: {s['current_price']} | 前日比: {s['price_change_pct']:+.1f}%
  出来高比: {s['volume_ratio']:.1f}倍 | RSI: {s['rsi']}
  クリア条件: {checks_str}
  セクター: {s['sector']}
"""
    prompt = f"""
今日（{datetime.now().strftime('%Y年%m月%d日')}）のスクリーニング結果です。
以下の銘柄について機関投資家目線で分析してください。

{stock_info}

各銘柄について：
1. 注目ポイント（1〜2文）
2. 主なリスク（1文）
3. 判定：★★★（強い買い候補）/ ★★（要確認）/ ★（見送り）

最後に「本日の総評」を2〜3文でまとめてください。
LINE通知用なので全体で500文字以内でお願いします。
"""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def send_line_notify(message):
    url = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    data = {"message": message}
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        print("✅ LINE通知送信成功！")
    else:
        print(f"❌ LINE通知失敗: {response.status_code} {response.text}")

def build_line_message(candidates, analysis):
    now = datetime.now().strftime("%m/%d %H:%M")
    header = f"\n📈 株式分析レポート {now}\n{'='*25}\n"
    if not candidates:
        return header + "\n本日の候補銘柄なし\n市場は様子見モードです。"
    summary = f"\n🎯 候補銘柄 {len(candidates[:5])}件\n"
    for i, s in enumerate(candidates[:5], 1):
        star = "🟢" if s["score"] == 4 else "🟡"
        summary += f"{star} {i}. {s['name']}\n"
        summary += f"   {s['current_price']} ({s['price_change_pct']:+.1f}%) RSI:{s['rsi']}\n"
    analysis_section = f"\n{'='*25}\n🤖 AI分析\n{analysis}"
    footer = f"\n{'='*25}\n⚠️ 投資は自己責任でお願いします"
    return header + summary + analysis_section + footer

def main():
    print("🚀 株式自動分析システム 起動")
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not LINE_TOKEN:
        print("❌ LINE_TOKEN が設定されていません")
        return
    if not ANTHROPIC_API_KEY:
        print("❌ ANTHROPIC_API_KEY が設定されていません")
        return
    candidates = screen_stocks(WATCHLIST)
    print(f"\n✅ スクリーニング完了: {len(candidates)}銘柄が候補")
    print("\n🤖 Claude分析中...")
    analysis = analyze_with_claude(candidates)
    message = build_line_message(candidates, analysis)
    print(message)
    send_line_notify(message)

if __name__ == "__main__":
    main()
