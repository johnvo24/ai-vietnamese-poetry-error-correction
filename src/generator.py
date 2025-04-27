class TextGenerator():
  def __init__(self, model, tokenizer, device, special_tokens, topk=5, max_tokens=256, maxlen=512):
    self.model = model
    self.tokenizer = tokenizer
    self.device = device
    self.special_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    self.k = topk
    self.max_tokens = max_tokens
    self.maxlen = maxlen

  def generate(self, start_tokens):
    pass