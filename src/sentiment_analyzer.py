import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["positive", "negative", "neutral"]

    def analyze_sentiment(self, headlines: list):
        if not headlines:
            return []

        # Filter out empty or non-string headlines
        processed_headlines = [h.strip() for h in headlines if isinstance(h, str) and h.strip()]
        if not processed_headlines:
            return [{"sentiment": "neutral", "confidence": 1.0, "error": "Empty or invalid headlines provided."}] * len(headlines)

        # Tokenize and predict in batches
        inputs = self.tokenizer(processed_headlines, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        results = []
        for i, headline in enumerate(headlines):
            if isinstance(headline, str) and headline.strip():
                # Find the corresponding prediction for the processed headline
                try:
                    idx = processed_headlines.index(headline.strip())
                    sentiment_scores = predictions[idx].numpy()
                    predicted_label_idx = np.argmax(sentiment_scores)
                    sentiment = self.labels[predicted_label_idx]
                    confidence = float(sentiment_scores[predicted_label_idx])
                    results.append({"sentiment": sentiment, "confidence": confidence})
                except ValueError:
                    # This case should ideally not happen if processed_headlines is correctly built
                    results.append({"sentiment": "neutral", "confidence": 0.0, "error": "Headline processing error."})
            else:
                results.append({"sentiment": "neutral", "confidence": 0.0, "error": "Empty or non-string headline."})
        
        return results

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_headlines = [
        "Stocks rally on strong earnings report.",
        "Company faces significant legal challenges.",
        "Market shows no clear direction.",
        "",
        "Another positive quarter for tech.",
        None,
        "The economy is in a recession.",
        "  "
    ]
    sentiments = analyzer.analyze_sentiment(sample_headlines)
    for i, s in enumerate(sentiments):
        print(f"Headline: '{sample_headlines[i]}' -> Sentiment: {s['sentiment']}, Confidence: {s['confidence']:.4f}")
