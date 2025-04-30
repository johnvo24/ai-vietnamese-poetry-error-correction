from src.dataset_handler import DatasetHandler
from src.models import VpecGPT2, VpecDeepSeek, VpecGemma3, VpecQwen3
from src import helper
from Jvai import GDrive

dataset_handler = DatasetHandler()

def train_sft(option="vpec_deepseek"):
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
  train_loader, val_loader, _ = dataset_handler.get_data_loader(tokenizer=vpec.tokenizer)
  for name, param in vpec.model.named_parameters():
    print(f"{name}: {param.dtype}")
  # Start training
  vpec.__train_sft__(
    train_loader=train_loader,
    val_loader=val_loader
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
  vpec.__generate__("<sop> Trời xanh soi mắt em xanh,\nBiển xanh con sóng cuộn nhanh xô bờ.\nEm ra biển ngắm tivi chờ,\nCâu thơ lục bát ngẩn ngơ biển chiều. <eop> <reasoning_memory> Tóm tắt ngữ cảnh: Bài thơ thể hiện nỗi cô đơn, buồn bã của một người chờ đợi trong tình yêu. <eois> Sửa lỗi RE: Thay ""ngẩn ngơ"" bằng ""ngẩn ngơ"" ở dòng 4 tại từ thứ 8 <eois>")


train_sft(option="vpec_qwen3")
# generate(option="vpec_deepseek", g_drive=True)

# vpec_deepseek = VpecDeepSeek()
# print(vpec_deepseek.model)