import torch
from transformers import XLNetTokenizer, XLNetModel
import pandas as pd
import numpy as np
from tqdm import tqdm

class XLNetFeatureExtractor:
    def __init__(self, model_name='xlnet-base-cased'):
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetModel.from_pretrained(model_name)
        self.model.eval() # Set to evaluation mode
        
    def tabular_to_text(self, df, feature_cols):
        """
        Converts tabular rows into text descriptors.
        Example: "Nitrogen is 80. Phosphorus is 40..."
        """
        texts = []
        for _, row in df.iterrows():
            text_parts = [f"{col} is {row[col]}" for col in feature_cols]
            texts.append(". ".join(text_parts))
        return texts
        
    def extract_features(self, texts, batch_size=32):
        """
        Passes text through XLNet to get the [CLS] token embedding or mean pooling.
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting XLNet Features"):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use the last hidden state average as feature representation
            # last_hidden_state shape: (batch, seq_len, hidden_size)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean pooling over the sequence dimension
            mean_embedding = torch.mean(last_hidden_state, dim=1).numpy()
            embeddings.append(mean_embedding)
            
        return np.vstack(embeddings)

if __name__ == "__main__":
    # Test run
    df = pd.DataFrame({'N': [10, 20], 'P': [5, 10], 'K': [15, 30]})
    extractor = XLNetFeatureExtractor()
    texts = extractor.tabular_to_text(df, ['N', 'P', 'K'])
    feats = extractor.extract_features(texts)
    print("Extracted shape:", feats.shape)
