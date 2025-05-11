import time
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import configs as config
from torch.optim import AdamW
from src.trainer import Trainer
from peft import PeftModel, LoraConfig, get_peft_model
from src import helper

class VpecQwen3():
  def __init__(self):
    self.model_name = config.VPEC_QWEN_MODEL_NAME
    self.model_id = 'Qwen/Qwen3-0.6B-Base'
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # self.device = torch.device('cpu')
    # Define special tokens
    self.special_tokens = [
      "<sep>", "<sop>", "<eop>", "<reasoning_memory>", "<error>", "<desc>", "<reason>",
      "<action>", "<replace>", "<line>", "<index>", "<effect>", "<eois>", "<eos>"
    ]
    # Initialize
    self.tokenizer = self._load_tokenizer()
    self.model = None
    # Create Optimizer
    self.optimizer = None

  def _load_tokenizer(self):
    tokenizer = AutoTokenizer.from_pretrained(self.model_id)
    tokenizer.add_special_tokens({
      'additional_special_tokens': self.special_tokens,
      'pad_token': '<pad>'
    })
    return tokenizer
  
  def _load_model(self):
    # Load configuration and adjust dropout settings
    model_config = AutoConfig.from_pretrained(self.model_id)
    model_config.attention_dropout = 0.1
    model_config.resid_pdrop = 0.1
    model_config.embd_pdrop = 0.1

    self.model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      config=model_config,
      device_map='auto',
      trust_remote_code=True
    ).to(self.device)
    
    self.model.resize_token_embeddings(len(self.tokenizer))
    self.model.config.pad_token_id = self.tokenizer.pad_token_id
    
    self.optimizer = AdamW(
      filter(lambda p: p.requires_grad, self.model.parameters()),
      lr=config.SFT_QWEN_LEARNING_RATE
    )
  
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
    self.model = get_peft_model(base_model, lora_config)
    self.optimizer = AdamW(
      filter(lambda p: p.requires_grad, self.model.parameters()), 
      lr=config.SFT_QWEN_LEARNING_RATE
    )

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
        log_dir=config.LOG_DIR + f"/{self.model_name}/run_{time.strftime('%Y%m%d_%H%M%S')}",
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS
      )
      trainer.train(config.SFT_QWEN_EPOCHS, start_epoch)
    except FileNotFoundError as e:
      print("[ERROR] FileNotFoundError: No such file: best_checkpoint.tar")

  def __generate__(self, input_text, num_return_sequences=1, max_target_length=None):
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
      eos_token_id=self.tokenizer.convert_tokens_to_ids('<eois>'),
      # num_beams=5,        # Beam Search with 5 beams
      do_sample=True,
      top_k=50,           # Top 50 best token
      top_p=0.9,          # Chooses the most probable tokens whose cumulative probability (xac suat tich luy) is at most 0.9
      temperature=0.5,    # Controls the creativity of the model
      num_return_sequences=num_return_sequences,
    )

    result = []
    for index in range(outputs.shape[0]):
      text_generated = outputs[index][inputs['input_ids'].shape[1]: ]
      output_text = self.tokenizer.decode(text_generated, skip_special_tokens=False)
      first_eos_index = output_text.find('<eos>')
      first_eois_index = output_text.find('<eois>')
      if first_eos_index != -1:
        reasoning_step = output_text[:first_eos_index + len("<eos>")].strip()
      elif first_eois_index != -1:
        reasoning_step = output_text[:first_eois_index + len("<eois>")].strip()
      else:
        reasoning_step = output_text.strip()
      # print("Reasoning Step: \n", reasoning_step)
      result.append(reasoning_step)
    return result