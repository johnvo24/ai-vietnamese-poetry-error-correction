from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from Jvai import JDataPreprocessor
from src.reasoning_dataset import ReasoningDataset
import configs as jconfig
  
class DatasetHandler():
  def __init__(self):
    self.dataset = None
    self.dataPreprocessor = JDataPreprocessor()

  def _load_data(self):
    # ??? Load dataset from Raw Dataset
    # data processing can be performed here
    self.dataset = self.dataPreprocessor.read_data(
      file_path=jconfig.SFT_DATASET_FILE_PATH,
      drop_duplicates_from=[],
      drop_column_value={}, 
      columns_selected=['error_poem', 'step_content', 'edited_poem'],
      read_size=None
    )
    print(f"[JV] SFT Gold Dataset: {self.dataset.shape}")
    return self.dataset
  
  def _is_within_max_length(self, row, tokenizer):
    input_tokens = tokenizer.encode(row['error_poem'])
    label_tokens = tokenizer.encode(row['step_content'])
    return (len(input_tokens) + len(label_tokens) + 1 <= jconfig.MAX_LENGTH) and (len(input_tokens) + 1 <= jconfig.MAX_INPUT_LENGTH) # 1 slot for <sep> token
  
  def _filter_reasoning_memory(self, sample):
    parts = sample.split('<eois>')
    if len(parts) <= jconfig.MAX_REASONING_MEMORY:
      return sample
    return '<eois>'.join(parts[:1] + parts[-(jconfig.MAX_REASONING_MEMORY):])
  
  def split_data(self, save_dataset=False, tokenizer=None):
    # ??? Split the data into training, evaluation and test sets
    # and then save them to files
    self._load_data()
    self.dataset['error_poem'] = self.dataset['error_poem'].apply(
      lambda sample: self._filter_reasoning_memory(sample)
    )
    # [print(len(tokenizer.encode(x))) for x in self.dataset['error_poem'].tolist()]

    if tokenizer:
      self.dataset = self.dataset[self.dataset.apply(lambda row: self._is_within_max_length(row=row, tokenizer=tokenizer), axis=1)].reset_index(drop=True)[:jconfig.SFT_DATASET_SIZE]
      print(f"[JV] SFT Gold Dataset For Training: {self.dataset.shape}")

    train_set, val_test_set = train_test_split(
      self.dataset, 
      test_size=(jconfig.SFT_VAL_SIZE+jconfig.SFT_TEST_SIZE)/100,
      random_state=jconfig.SFT_RANDOM_STATE,
      shuffle=jconfig.SFT_SPLIT_WITH_SHUFFLE
    )
    val_set, test_set = train_test_split(
      val_test_set, 
      test_size=jconfig.SFT_TEST_SIZE/(jconfig.SFT_VAL_SIZE+jconfig.SFT_TEST_SIZE),
      random_state=jconfig.SFT_RANDOM_STATE,
      shuffle=jconfig.SFT_SPLIT_WITH_SHUFFLE
    )
    print(f"[JV] Splitted Dataset:\n> total: {self.dataset.shape}\n> train_set: {train_set.shape}\n> val_set: {val_set.shape}\n> test_set: {test_set.shape}")
    # Save to file
    if save_dataset:
      train_set.to_csv(jconfig.SFT_TRAIN_DATASET_PATH, index=False)
      val_set.to_csv(jconfig.SFT_VAL_DATASET_PATH, index=False)
      test_set.to_csv(jconfig.SFT_TEST_DATASET_PATH, index=False)
    print(f"[JV] Saved training, validation and test sets to files.")

  def get_data_loader(self, tokenizer):
    train_set = self.dataPreprocessor.read_data(file_path=jconfig.SFT_TRAIN_DATASET_PATH)
    val_set = self.dataPreprocessor.read_data(file_path=jconfig.SFT_VAL_DATASET_PATH)
    test_set = self.dataPreprocessor.read_data(file_path=jconfig.SFT_TEST_DATASET_PATH)

    train_dataset = ReasoningDataset(dataframe=train_set, tokenizer=tokenizer, max_length=jconfig.MAX_LENGTH)
    val_dataset = ReasoningDataset(dataframe=val_set, tokenizer=tokenizer, max_length=jconfig.MAX_LENGTH)
    test_dataset = ReasoningDataset(dataframe=test_set, tokenizer=tokenizer, max_length=jconfig.MAX_LENGTH)

    # Attach pin_memory=True if training in GPU
    train_loader = DataLoader(dataset=train_dataset, batch_size=jconfig.BATCH_SIZE, shuffle=jconfig.SHUFFLE, num_workers=jconfig.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=jconfig.BATCH_SIZE, shuffle=jconfig.SHUFFLE, num_workers=jconfig.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=jconfig.BATCH_SIZE, shuffle=jconfig.SHUFFLE, num_workers=jconfig.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader