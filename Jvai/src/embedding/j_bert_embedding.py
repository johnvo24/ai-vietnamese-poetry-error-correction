import torch
import numpy as np
from tqdm import tqdm

from transformers import BertTokenizer, BertModel

class BertEmbedder():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bertModel: BertModel = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.encoded_inputs = {}
        self.features = torch.empty(0, 0, 0)

    # Tokenization ------------------------------------------< JOHN TAG
    def tokenize_plus(self, texts: list):
        print(f"[JV] ==========< TOKENIZATION >==========")
        len_texts = [len(text) for text in texts]
        print(f"[JV] Length => max: {np.max(len_texts)}, min: {np.min(len_texts)}, mean: {np.mean(len_texts)}, median: {np.median(len_texts)}")
        self.encoded_data = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True, 
            return_attention_mask=True, 
            pad_to_max_length=True, 
            max_length=512,
            return_tensors='pt'
        )

    def get_embedding(self):
        self.bertModel.eval()

        print(f"[JV] Model is running on: {next(self.bertModel.parameters()).device}")
        batch_size = 32
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(self.encoded_data['input_ids']), batch_size), desc="Calculating Embeddings"):
                batch_input_ids = self.encoded_data['input_ids'][i:i + batch_size].to(self.device)
                batch_attention_mask = self.encoded_data['attention_mask'][i:i + batch_size].to(self.device)
                
                outputs = self.bertModel(batch_input_ids, attention_mask=batch_attention_mask)
                batch_last_hidden_states = outputs.last_hidden_state
                embeddings.append(batch_last_hidden_states.cpu()) # Switch back to cpu before concatenating
        
        self.features = torch.cat(embeddings, dim=0)
        print(f"[JV] Shape of Last hidden states: {self.features.shape}")