from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import configs as config
from torch.optim import AdamW
from src.trainer import Trainer
from src import helper

class VpecGPT2():
  def __init__(self):
    self.model_name = config.VPEC_GPT2_MODEL_NAME
    self.model_id = 'gpt2'
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.special_tokens = [
      "<sep>", "<sop>", "<eop>", "<reasoning_memory>", "<error>", "<desc>", "<reason>",
      "<action>", "<replace>", "<line>", "<index>", "<effect>", "<eois>", "<eos>"
    ]
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    self.tokenizer.add_special_tokens({
      'additional_special_tokens': self.special_tokens,
      'pad_token': '<pad>'
    })
    self.model = AutoModelForCausalLM.from_pretrained('gpt2').to(self.device)
    self.model.resize_token_embeddings(len(self.tokenizer))
    self.model.config.pad_token_id = self.tokenizer.pad_token_id
    self.model = self.model.to(self.device)
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

  def __generate__(self, input_text, max_target_length=None):
    inputs = self.tokenizer(
      input_text + '<sep>',
      padding=False,
      truncation=True,
      max_length=config.MAX_INPUT_LENGTH,
      return_tensors="pt"
    ).to(self.device)
    print(self.tokenizer.decode(inputs["input_ids"][0]))
    outputs = self.model.generate(
      input_ids=inputs['input_ids'],
      attention_mask=inputs['attention_mask'],
      max_length=inputs['input_ids'].shape[1] + max_target_length if max_target_length else config.MAX_LENGTH,
      eos_token_id=[self.tokenizer.convert_tokens_to_ids('<eois>'), self.tokenizer.convert_tokens_to_ids('<eos>')],
      num_beams=5,
      early_stopping=True
    )
    text_generated = outputs[0][inputs['input_ids'].shape[1]: ]
    result = self.tokenizer.decode(text_generated, skip_special_tokens=False)
    print("Reasoning Step: \n", result)