from src.dataset_handler import DatasetHandler

from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to("cuda")

special_tokens = [
  "<sop>", "<eop>", "<reasoning_memory>", "<error>", "<desc>", "<reason>",
  "<action>", "<replace>", "<line>", "<index>", "<effect>", "<eois>", "<eos>"
]
print(tokenizer.vocab_size)
tokenizer.add_special_tokens({
  'additional_special_tokens': special_tokens,
  'pad_token': '<pad>'
})
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

dataset_handler = DatasetHandler()
dataset_handler.split_data(save_dataset=True, tokenizer=tokenizer)
train_loader, val_loader, test_loader = dataset_handler.get_data_loader(tokenizer=tokenizer)
# for batch in train_loader:
#   for key in batch:
#     print(batch[key])