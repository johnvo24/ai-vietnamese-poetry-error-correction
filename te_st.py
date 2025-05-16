import pandas as pd
from src.models import VpecGPT2, VpecDeepSeek, VpecGemma3, VpecQwen3
from src import helper
from tqdm import tqdm
from Jvai import GDrive
import os

def save_model_to_gdrive(model_name):
  helper.save_best_checkpoint_to_gdrive(model_name)

def test_model(g_drive=False):
  vpec = VpecQwen3()
  vpec._load_model()
  if g_drive:
    checkpoint = GDrive().load_model_from_drive('best_checkpoint.tar', vpec.model_name)
    vpec.model.load_state_dict(checkpoint['model_state_dict'])
    vpec.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  else:
    checkpoint = helper.load_checkpoint(
      model_dir=vpec.model_name,
      model=vpec.model,
      optimizer=vpec.optimizer,
      is_the_best=True,
    )
    vpec.model = checkpoint['model']
    vpec.optimizer = checkpoint['optimizer']

  df = pd.read_csv('data/sft_dataset/gold_cot_data/test_dataset.csv')
  sequence_per_sample = 5
  result = []

  for index, sample in tqdm(df.iterrows(), total=len(df), desc="Generating reasoning step"):
    result += vpec.__generate__(input_text=sample['error_poem'], num_return_sequences=sequence_per_sample)
    print(len(result))

  data = pd.DataFrame(result)
  helper.makedir('data', 'generated_data')
  data.to_csv('data/generated_data/test_data.csv', index=False)
  gdrive = GDrive()
  gdrive.upload_file(
    file_name='test_data.csv',
    file_path=os.path.join('data', 'generated_data/test_data.csv'),
    folder_path='/generated_data'
  )

# save_model_to_gdrive(model_name="vpec_qwen3")
# save_model_to_gdrive(model_name="vpec_qwen3_0")
test_model(g_drive=True)