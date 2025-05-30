from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class ParsBERTSentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-zwnj-base-sentiment-snappfood")
        self.model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-zwnj-base-sentiment-snappfood")

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        sentiment = torch.argmax(probs).item()
        confidence = probs[0][sentiment].item()
        labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return labels[sentiment], confidence