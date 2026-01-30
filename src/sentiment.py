import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class FinBertAnalyzer:
    def __init__(self):
        print("Loading FinBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, text_list):
        # Transformer-specific Tokenization
        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
        # ProsusAI/finbert Mapping: [Positive, Negative, Neutral]
        # Scalar Score = Prob(Positive) - Prob(Negative)
        scalar_scores = probs[:, 0] - probs[:, 1] 
        return scalar_scores

def apply_sentiment_analysis(df, column_name='Headline'):
    """
    Apply sentiment on RAW text to preserve context, 
    """
    analyzer = FinBertAnalyzer()
    batch_size = 16
    results = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df[column_name].iloc[i:i+batch_size].astype(str).tolist()
        results.extend(analyzer.predict(batch))
        
    df['sentiment_score'] = results
    return df