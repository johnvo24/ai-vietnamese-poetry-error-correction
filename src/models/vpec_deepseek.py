import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import configs as config
from torch.optim import AdamW
from src.trainer import Trainer
from peft import PeftModel, LoraConfig, get_peft_model
from src import helper

class VpecDeepSeek():
  def __init__(self):
    self.model_name = config.VPEC_DEEPSEEK_MODEL_NAME
    self.model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.device = torch.device('cpu')
    # Define special tokens
    self.special_tokens = [
      "<sep>", "<sop>", "<eop>", "<reasoning_memory>", "<error>", "<desc>", "<reason>",
      "<action>", "<replace>", "<line>", "<index>", "<effect>", "<eois>", "<eos>"
    ]
    # Initialize
    self.tokenizer = self._load_tokenizer()
    self.model = self._load_model_with_qlora()
    # Create Optimizer
    self.optimizer = AdamW(
      filter(lambda p: p.requires_grad, self.model.parameters()), 
      lr=config.SFT_VPECDEEPSEEK_LEARNING_RATE
    )

  def _load_tokenizer(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    tokenizer.add_special_tokens({
      'additional_special_tokens': self.special_tokens,
      'pad_token': '<pad>'
    })
    return tokenizer
  
  def _load_model_with_qlora(self):
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      quantization_config=bnb_config,
      device_map='auto',
      trust_remote_code=True
    ).to(self.device)
    base_model.resize_token_embeddings(len(self.tokenizer))
    base_model.config.pad_token_id = self.tokenizer.pad_token_id
    lora_config = LoraConfig(
      r=8,                    # Rank for LoRA
      lora_alpha=32,          # Alpha value
      lora_dropout=0.5,       # Dropout for LoRA
      bias="none",            # Bias term in LoRA
      task_type="CAUSAL_LM",  # Causal LM task
      # target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]  # Module to apply LoRA
      target_modules=["self_attn.q_proj", "self_attn.v_proj", "mlp.down_proj"]  # Module to apply LoRA
    )
    model = get_peft_model(base_model, lora_config)
    return model

  def __train_sft__(self, train_loader, val_loader, from_best_checkpoint=False):
    try:
      start_epoch = -1
      if from_best_checkpoint:
        checkpoint = helper.load_checkpoint(self.model_name, self.model, self.optimizer, is_the_best=True)
        start_epoch = checkpoint['epoch']
        print(f"[JV] Loading model from {self.model_name}/best_checkpoint.tar [EPOCH: {start_epoch}]")
        self.model = checkpoint['model']
        self.optimizer = checkpoint['optimizer']

      trainer = Trainer(
        model=self.model,
        model_dir_name=self.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=self.optimizer,
        log_dir=config.LOG_DIR + f"/{self.model_name}/run_{time.strftime('%Y%m%d_%H%M%S')}"
      )
      trainer.train(config.SFT_VPECDEEPSEEK_EPOCHS, start_epoch)
    except FileNotFoundError as e:
      print("[ERROR] FileNotFoundError: No such file: best_checkpoint.tar")

  def __generate__(self, input_text, max_target_length=None):
    inputs = self.tokenizer(
      input_text + '<sep>',
      padding=False,
      truncation=True,
      max_length=config.MAX_INPUT_LENGTH,
      return_tensors="pt"
    ).to(self.device)
    outputs = self.model.generate(
      input_ids=inputs['input_ids'],
      attention_mask=inputs['attention_mask'],
      max_length=inputs['input_ids'].shape[1] + max_target_length if max_target_length else config.MAX_LENGTH,
      eos_token_id=[self.tokenizer.convert_tokens_to_ids('<eois>'), self.tokenizer.convert_tokens_to_ids('<eos>')],
      num_beams=5,        # Beam Search with 5 beams
      top_k=50,           # Top 50 best token
      top_p=0.9,          # Chooses the most probable tokens whose cumulative probability (xac suat tich luy) is at most 0.9
      temperature=0.7,    # Controls the creativity of the model
      early_stopping=True
    )
    text_generated = outputs[0][inputs['input_ids'].shape[1]: ]
    result = self.tokenizer.decode(text_generated, skip_special_tokens=False)
    print("Reasoning Step: \n", result)