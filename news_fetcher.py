import feedparser
from datetime import datetime, timezone, timedelta

def get_farsi_economic_news():
    rss_urls = [
        "https://www.eghtesadonline.com/rss",
        "https://www.mehrnews.com/rss/fa/economy",
    ]
    articles = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:
            # تبدیل تاریخ به datetime
            if hasattr(entry, 'published_parsed'):
                published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc) + timedelta(hours=3, minutes=30)  # تهران +3:30
            else:
                published = datetime.now()
            articles.append({
                "title": entry.title,
                "summary": entry.summary,
                "link": entry.link,
                "published": published
            })
    return articles