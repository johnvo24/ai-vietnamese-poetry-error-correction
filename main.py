from src.dataset_handler import DatasetHandler
from src.models import VpecGPT2
from src import helper

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

def generate_vpec_gpt2():
  vpec_gpt2 = VpecGPT2()
  checkpoint = helper.load_checkpoint(
    model_dir=vpec_gpt2.model_name,
    model=vpec_gpt2.model,
    optimizer=vpec_gpt2.optimizer,
    is_the_best=True
  )
  vpec_gpt2.model = checkpoint['model']
  vpec_gpt2.optimizer = checkpoint['optimizer']
  vpec_gpt2.__generate__("<sop> Con cò mà đi ăn đêm\nĐậu phải cành đào lộn cổ xuống ao <eop> <reasoning_memory> Bất kể mọi lúc đều có thể có rủi ro. <eois>", 256)

train_vpec_gpt2()
# generate_vpec_gpt2()