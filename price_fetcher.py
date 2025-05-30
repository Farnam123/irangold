import yfinance as yf
import requests

def get_gold_price():
    data = yf.download("GC=F", period="7d", interval="5m")
    if data.empty:
        return None
    return data

def get_dollar_price():
    url = "https://www.tgju.org/api/commodities"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            data = res.json()
            dollar = next((item for item in data if item['name'] == 'دلار'), None)
            if dollar:
                return float(dollar['price'])
    except Exception as e:
        print("Error fetching dollar price:", e)
    return None