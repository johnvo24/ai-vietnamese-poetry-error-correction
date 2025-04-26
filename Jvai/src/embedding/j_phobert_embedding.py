import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class PhoBertEmbedder():
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

  def tokenize_plus(self, texts: list):
    print(f"[JV] ==========< TOKENIZATION >==========")
    len_texts = [len(text) for text in texts]
    print(f"[JV] Length => max: {np.max(len_texts)}, min: {np.min(len_texts)}, mean: {np.mean(len_texts)}, median: {np.median(len_texts)}")
    return self.tokenizer.batch_encode_plus(
      texts,
      add_special_tokens=True,
      return_attention_mask=True,
      padding='max_length',
      max_length=256,
      truncation=True,
      return_tensors='pt'
    )
  
  def get_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
      self.model.eval()
      print((input_ids.shape, attention_mask.shape))
      batch_size = 32
      embeddings = []
      with torch.no_grad():
          for i in tqdm(range(0, len(input_ids), batch_size), desc="Calculating Embeddings"):
              batch_input_ids = input_ids[i:i + batch_size].to(self.device)
              batch_attention_mask = attention_mask[i:i + batch_size].to(self.device)
              
              outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask)
              batch_last_hidden_states = outputs.last_hidden_state
              embeddings.append(batch_last_hidden_states.cpu()) # Switch back to cpu before concatenating
      embeddings = torch.cat(embeddings, dim=0)
      return embeddings

# embedder = PhoBertEmbedder()
# encoded_inputs = embedder.tokenize_plus(["Những ước mơ hoài bão của bạn sẽ la", "Những ước mơ hoài bão của bạn sẽ la"])
# # print(encoded_inputs)
# print(embedder.get_embedding(encoded_inputs['input_ids'], encoded_inputs['attention_mask']))