from src.dataset_handler import DatasetHandler
from src.models import VpecGPT2, VpecDeepSeek, VpecGemma3, VpecQwen3
from src import helper
from Jvai import GDrive

dataset_handler = DatasetHandler()

def prepare_dataset(option="vpec_deepseek"):
  if option == "vpec_gpt2":
    vpec = VpecGPT2()
  elif option == "vpec_deepseek":
    vpec = VpecDeepSeek()
  elif option == "vpec_gemma3":
    vpec = VpecGemma3()
  elif option == "vpec_qwen3":
    vpec = VpecQwen3()
  else:
    print("What a stupid option!")
    exit(0)
  dataset_handler.split_data(save_dataset=True, tokenizer=vpec.tokenizer)
  print("[JV] Dataset preparation done!")

def train_sft(option="vpec_deepseek", from_best_checkpoint=False):
  if option == "vpec_gpt2":
    vpec = VpecGPT2()
  elif option == "vpec_deepseek":
    vpec = VpecDeepSeek()
  elif option == "vpec_gemma3":
    vpec = VpecGemma3()
  elif option == "vpec_qwen3":
    vpec = VpecQwen3()
    vpec._load_model()
  else:
    print("What a stupid option!")
    exit(0)
  train_loader, val_loader, _ = dataset_handler.get_data_loader(tokenizer=vpec.tokenizer)
  # for name, param in vpec.model.named_parameters():
  #   print(f"{name}: {param.dtype}")
  # Start training
  vpec.__train_sft__(
    train_loader=train_loader,
    val_loader=val_loader,
    from_best_checkpoint=from_best_checkpoint
  )

def generate(option="vpec_deepseek", g_drive=False):
  if option == "vpec_gpt2":
    vpec = VpecGPT2()
  elif option == "vpec_deepseek":
    vpec = VpecDeepSeek()
  elif option == "vpec_gemma3":
    vpec = VpecGemma3()
  elif option == "vpec_qwen3":
    vpec = VpecQwen3()
    vpec._load_model()
  else:
    print("What a stupid option!")
    exit(0)

  if g_drive:
    checkpoint = GDrive().load_model_from_drive('best_checkpoint.tar', vpec.model_name)
    vpec.model.load_state_dict(checkpoint['model_state_dict'])
    vpec.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  else:
    checkpoint = helper.load_checkpoint(
      model_dir=vpec.model_name,
      model=vpec.model,
      optimizer=vpec.optimizer,
      is_the_best=True
    )
    vpec.model = checkpoint['model']
    vpec.optimizer = checkpoint['optimizer']
  result = vpec.__generate__(input_texts=["<sop> Con cò mà đi ăn đêm\nĐi giữa dọc đường rơi tủm. <eop> <reasoning_memory> Tóm tắt ngữ cảnh: Bài thơ thể hiện sự cô đơn, lạc lõng và những điều giản dị, đời thường khiến con người cảm thấy trống trải <eois>"], num_return_sequences=5)
  print(result)

def main():
  while True:
    print("Options:\n1.vpec_qwen3\n2.vpec_gp2\n3. vpec_deepseek\n4.vpec_gemma3")
    inp = int(input("Your choice: "))
    if inp < 1 or inp > 4:
      continue
    else:
      option = "vpec_qwen3" if inp == 1 else "vpec_gpt2" if inp == 2 else "vpec_deepseek" if inp == 3 else "vpec_gemma3"
      break
  while True:
    print("Methods:\n1.Preparing\n2.Training\n3. Generating")
    inp1 = int(input("Your choice: "))
    if inp1 < 1 or inp1 > 3:
      continue
    else:
      break
  if inp1 == 1: prepare_dataset(option=option)
  elif inp1 == 2: train_sft(option=option, from_best_checkpoint=True)
  elif inp1 == 3: generate(option=option, g_drive=False)

if __name__ == "__main__":
  main()
