import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import config

class GPT2Encoder():
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.add_special_tokens({
      'pad_token': '<|pad|>', # padding
      'eos_token': '<|eos|>', # end_of_text
    })
    self.model.resize_token_embeddings(len(self.tokenizer))

  def encode(self, prompts: list, labels: list):
    # prompt: describe the text to be created
    # label: the corresponding label(poem) for the prompt

    labels = [label + '<|eos|>' for label in labels]
    print(labels)

    prompt_encodings = self.tokenizer(
      prompts,
      truncation=True,
      padding=True,
      max_length=512,
      return_tensors='pt'
    )
    labels_encodings = self.tokenizer(
      labels,
      truncation=True,
      padding=True,
      max_length=512,
      return_tensors='pt'
    )
    return {
      'input_ids': prompt_encodings['input_ids'],
      'attention_mask': prompt_encodings['attention_mask'],
      'labels': labels_encodings['input_ids'],
      'labels_attention_mask': labels_encodings['attention_mask']
    }


# max_length
# train model
# change exporting file