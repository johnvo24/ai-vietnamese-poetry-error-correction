from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import configs as config
from torch.optim import AdamW
from src.trainer import Trainer

class VpecGPT2():
  def __init__(self):
    self.model_name = config.VPEC_GPT2_MODEL_NAME
    self.model_id = 'gpt2'
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.special_tokens = [
      "<sop>", "<eop>", "<reasoning_memory>", "<error>", "<desc>", "<reason>",
      "<action>", "<replace>", "<line>", "<index>", "<effect>", "<eois>", "<eos>"
    ]
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    self.configuration = GPT2Config(vocab_size=len(self.tokenizer), n_layer=8)
    self.tokenizer.add_special_tokens({
      'additional_special_tokens': self.special_tokens,
      'pad_token': '<pad>'
    })
    self.model = GPT2LMHeadModel(self.configuration).to(self.device)
    self.model.resize_token_embeddings(len(self.tokenizer))

    self.optimizer = AdamW(self.model.parameters(), config.SFT_VPECGPT2_LEARNING_RATE)

  def __train_sft__(self, train_loader, val_loader):
    trainer = Trainer(
      model=self.model,
      model_dir_name=self.model_name,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=self.optimizer,
      log_dir=config.LOG_DIR + f"/{self.model_name}"
    )
    trainer.train(config.SFT_VPECGPT2_EPOCHS)

  def __generate__(self, input_text):
    pass