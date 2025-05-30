from flask import Flask, render_template
from sentiment_analysis import ParsBERTSentiment
from news_fetcher import get_farsi_economic_news
from price_fetcher import get_gold_price, get_dollar_price
from dataset_builder import build_dataset
from model import train_model, GoldPricePredictor
import torch
import threading

app = Flask(__name__)

sentiment_model = ParsBERTSentiment()
model = None

def train_thread():
    global model
    print("شروع آموزش مدل ...")
    price_df = get_gold_price()
    news_list = get_farsi_economic_news()
    features, labels = build_dataset(price_df, news_list, sentiment_model)
    model = train_model(features, labels, epochs=10)
    print("آموزش مدل تمام شد.")

@app.route('/')
def index():
    gold_price_df = get_gold_price()
    dollar_price = get_dollar_price()
    news = get_farsi_economic_news()

    # تحلیل احساسات اخبار (5 خبر اول)
    sentiments = []
    for article in news[:5]:
        text = article['title'] + " " + article['summary']
        label, conf = sentiment_model.predict(text)
        sentiments.append({
            "title": article['title'],
            "sentiment": label,
            "confidence": round(conf, 2),
            "link": article['link']
        })

    latest_price = gold_price_df['Close'][-1] if gold_price_df is not None else None

    # پیش‌بینی با مدل (اگر آموزش داده شده)
    signal = "آموزش مدل در حال اجرا است."
    if model is not None and gold_price_df is not None:
        with torch.no_grad():
            # داده‌های آخر 10 سطر به عنوان ورودی مدل
            last_data = gold_price_df[['Close', 'MACD', 'RSI', 'EMA']].tail(10)
            # باید Sentiment هم اضافه شود، پس ساخت دیتاست با مقدار صفر موقتی
            last_data['Sentiment'] = 0
            x_input = torch.tensor(last_data.values).unsqueeze(0).float()
            pred = model(x_input).item()
            if pred > 0.6:
                signal = "سیگنال خرید"
            elif pred < 0.4:
                signal = "سیگنال فروش"
            else:
                signal = "نگهداری"

    return render_template('index.html', gold_price=latest_price, dollar_price=dollar_price,
                           news=sentiments, signal=signal)

if __name__ == '__main__':
    threading.Thread(target=train_thread).start()
    app.run(debug=True)