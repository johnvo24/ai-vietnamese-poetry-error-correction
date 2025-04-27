from src.dataset_handler import DatasetHandler
from src.models import VpecGPT2

dataset_handler = DatasetHandler()


def train_vpec_gpt2():
  vpec_gpt2 = VpecGPT2()
  dataset_handler.split_data(save_dataset=True, tokenizer=vpec_gpt2.tokenizer)
  train_loader, val_loader, test_loader = dataset_handler.get_data_loader(tokenizer=vpec_gpt2.tokenizer)
  # Start training
  vpec_gpt2.__train_sft__(
    train_loader=train_loader,
    val_loader=val_loader
  )

train_vpec_gpt2()