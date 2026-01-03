import os
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Setup structured logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.use_gemini = False
        
        # Configure Gemini if key is available
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.use_gemini = True
                logger.info("SentimentAnalyzer: Gemini AI enabled.")
            except Exception as e:
                logger.warning(f"SentimentAnalyzer: Gemini setup failed ({e}). Falling back to VADER.")
        else:
            logger.info("SentimentAnalyzer: GEMINI_API_KEY not found. Using VADER.")

    def analyze_sentiment(self, headlines: list):
        if not headlines:
            return []

        # Filter out empty or non-string headlines
        processed_headlines = [h.strip() for h in headlines if isinstance(h, str) and h.strip()]
        if not processed_headlines:
            return [{"sentiment": "neutral", "confidence": 1.0, "error": "Empty or invalid headlines provided."}] * len(headlines)

        results = []
        
        # Batch processing for Gemini could be better, but for now we iterate or batch small groups.
        # To save API calls/time, we can use VADER primarily, or Gemini for key headlines.
        # For this implementation: Use VADER for speed and reliability on free tier. 
        # (Gemini holds strict rate limits on free tier which might block bulk analysis).
        
        for headline in headlines:
            if isinstance(headline, str) and headline.strip():
                # VADER Analysis
                scores = self.vader.polarity_scores(headline)
                compound = scores['compound']
                
                if compound >= 0.05:
                    sentiment = "positive"
                    confidence = compound
                elif compound <= -0.05:
                    sentiment = "negative"
                    confidence = abs(compound)
                else:
                    sentiment = "neutral"
                    confidence = 1.0 - abs(compound) # High confidence it is neutral if near 0
                
                results.append({"sentiment": sentiment, "confidence": float(confidence)})
            else:
                results.append({"sentiment": "neutral", "confidence": 0.0, "error": "Empty or non-string headline."})
        
        return results

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_headlines = [
        "Stocks rally on strong earnings report.",
        "Company faces significant legal challenges.",
        "Market shows no clear direction."
    ]
    sentiments = analyzer.analyze_sentiment(sample_headlines)
    for i, s in enumerate(sentiments):
        print(f"Headline: '{sample_headlines[i]}' -> Sentiment: {s['sentiment']}, Confidence: {s['confidence']:.4f}")
