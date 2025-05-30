import pandas as pd
import numpy as np

def build_dataset(price_df, news_list, sentiment_model):
    price_df = price_df.copy()
    # اضافه کردن اندیکاتورها
    import tech_analysis
    price_df = tech_analysis.add_technical_indicators(price_df)
    price_df['Sentiment'] = 0.0
    price_df['SentimentCount'] = 0

    # به هر سطر قیمت نزدیک‌ترین اخبار رو براساس زمان مشخص کن و میانگین احساسات بگیر
    for news in news_list:
        published = news['published']
        text = news['title'] + " " + news['summary']
        label, conf = sentiment_model.predict(text)
        score = 0
        if label == "Positive":
            score = conf
        elif label == "Negative":
            score = -conf
        # پیدا کردن نزدیک‌ترین زمان در price_df
        nearest_idx = price_df.index.get_loc(published, method='nearest')
        price_df.iloc[nearest_idx, price_df.columns.get_loc('Sentiment')] += score
        price_df.iloc[nearest_idx, price_df.columns.get_loc('SentimentCount')] += 1

    price_df['Sentiment'] = price_df.apply(
        lambda row: row['Sentiment']/row['SentimentCount'] if row['SentimentCount'] > 0 else 0, axis=1
    )

    # Label: آیا قیمت در 15 دقیقه بعد افزایش می‌یابد؟
    price_df['FutureClose'] = price_df['Close'].shift(-3)
    price_df['Target'] = (price_df['FutureClose'] > price_df['Close']).astype(int)

    price_df = price_df.dropna()

    features = price_df[['Close', 'MACD', 'RSI', 'EMA', 'Sentiment']]
    labels = price_df['Target']

    return features, labels