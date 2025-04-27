from torch.utils.data.dataset import Dataset

class ReasoningDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_length):
    self.tokenizer = tokenizer
    self.dataframe = dataframe
    self.max_length = max_length

  def __len__(self):
    return len(self.dataframe)
  
  def __getitem__(self, index):
    input_text = self.dataframe.iloc[index]['error_poem']
    label_text = self.dataframe.iloc[index]['step_content']
    
    input_encoding = self.tokenizer(
      input_text,
      padding="max_length",
      truncation=True,
      max_length=self.max_length,
      return_tensors="pt"
    )
    label_encoding = self.tokenizer(
      label_text,
      padding="max_length",
      truncation=True,
      max_length=self.max_length,
      return_tensors="pt"
    )
    labels = label_encoding['input_ids'].squeeze()
    labels[labels == self.tokenizer.pad_token_id] = -100
    return {
      'input_ids': input_encoding['input_ids'].squeeze(), # token id
      'attention_mask': input_encoding['attention_mask'].squeeze(), # determine real token and padding token
      'labels': labels # use to calculate loss with output logit
    }