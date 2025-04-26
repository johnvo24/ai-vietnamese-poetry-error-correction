from transformers import RobertaTokenizerFast, AutoTokenizer, GPT2Tokenizer, BertTokenizer, DebertaTokenizer

class JTokenizer(object):
  def __init__(self):
    self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2") # Khả thi nhất
    self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    self.vibert_tokenizer = BertTokenizer.from_pretrained("FPTAI/vibert-base-cased")
    self.vnbert_tokenizer = BertTokenizer.from_pretrained("nguyenthanhasia/VNBertLaw")

  def phobert_tokenize(self, texts: list):
    encodings = self.phobert_tokenizer(
      text=texts, # 1st text
      # text_pair= , # 2nd text (optional: exp question & answer)
      add_special_tokens=True,
      padding="max_length",
      truncation=True, # truncate input string to max_length
      max_length=64,
      return_tensors="pt",
      return_token_type_ids=True,
      return_attention_mask=True,
    )
    return encodings
  
  def roberta_tokenize(self, texts: list):
    encodings = self.roberta_tokenizer(
      text=texts, # 1st text
      # text_pair= , # 2nd text (optional: exp question & answer)
      add_special_tokens=True,
      padding="max_length",
      truncation=True, # truncate input string to max_length
      max_length=64,
      return_tensors="pt",
      return_token_type_ids=True,
      return_attention_mask=True,
    )
    return encodings
    

# text = "Vào nhà, tôi đã chuẩn bị xong hết những món ngon cho mình và gia đình. Đường lối ngoằn ngoèo."
# # text = "Vao nha, toi da chuan bi xong het nhung mon ngon cho minh va gia dinh. Duong loi ngoan ngoeo."

# tokenizer = JTokenizer()
# encodings = tokenizer.phobert_tokenize([text, text])
# print(encodings)
# print(tokenizer.phobert_tokenizer.decode(encodings[0]["input_ids"]))
# print(tokenizer.phobert_tokenizer.tokenize(text))