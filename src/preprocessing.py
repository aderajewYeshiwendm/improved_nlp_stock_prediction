import re
import spacy
import os


class FinancialPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text):
        """Standard cleaning: URLs, special chars, extra whitespace."""
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Keep $, %, and numbers
        text = re.sub(r'[^\w\s%$.,]', '', text)
        return " ".join(text.split())

    def process_document(self, text):
        """
        Tokenization & Lemmatization.
        """
        clean_raw = self.clean_text(text)
        doc = self.nlp(clean_raw)
        
        # Tokenization & Lemmatization loop
        processed_tokens = []
        for token in doc:
            # Preserve Numberness: Keep currency and digits as-is
            if token.like_num or token.pos_ == "SYM":
                processed_tokens.append(token.text)
            elif not token.is_stop and not token.is_punct:
                processed_tokens.append(token.lemma_)
                
        return " ".join(processed_tokens)

def run_preprocessing_pipeline(df, text_column='Headline'):
    preprocessor = FinancialPreprocessor()
    print("Applying NLP Tokenization & Lemmatization...")
    df['processed_headline'] = df[text_column].apply(preprocessor.process_document)
    return df