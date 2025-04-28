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
    target_text = self.dataframe.iloc[index]['step_content']
    
    full_text = input_text + '<sep>' + target_text
    encoding = self.tokenizer(
      full_text,
      padding="max_length",
      truncation=True,
      max_length=self.max_length,
      return_tensors="pt"
    )
    
    input_ids = encoding['input_ids'].squeeze()
    attention_mask = encoding['attention_mask'].squeeze()
    labels = input_ids.clone()
    sep_token_index = input_ids.tolist().index(self.tokenizer.encode('<sep>')[0]) # Get separate token index
    labels[:sep_token_index + 1] = -100 # Mask input and <sep> 
    return {
      'input_ids': input_ids, # token id
      'attention_mask': attention_mask, # determine real token and padding token
      'labels': labels # use to calculate loss with output logit
    }