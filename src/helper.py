import os
import torch
import configs as config
from Jvai import GDrive

def device():
  return "cuda" if torch.cuda.is_available() else "cpu"

def makedir(path, name):
  dir_path = os.path.join(path, name)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"[JV] Created folder '{name}' at {path}")

def save_best_checkpoint_to_gdrive(model_dir):
  if config.USE_GDRIVE:
    old_file_id = GDrive().get_file_id('best_checkpoint.tar', f"/models/{model_dir}")
    GDrive().upload_file(
      file_name='best_checkpoint.tar',
      file_path=os.path.join(config.CHECKPOINT_DIR, f"{model_dir}/best_checkpoint.tar"),
      folder_path=f"/models/{model_dir}"
    )
    # Save new checkpoint before deleting old one
    if old_file_id:
      GDrive().delete_files([old_file_id])
    print(f"[JV] Saved checkpoint to GDrive at {model_dir}\\")

def save_checkpoint(model_dir, epoch, model, optimizer, is_the_best=False):
  makedir(config.CHECKPOINT_DIR, model_dir) # Create model_dir
  if is_the_best: # Save with name best_checkpoint
    file_path = os.path.join(config.CHECKPOINT_DIR, f"{model_dir}/best_checkpoint.tar")
  else: # Save with name checkpoint_epoch
    file_path = os.path.join(config.CHECKPOINT_DIR, f"{model_dir}/checkpoint_{epoch}.tar")
  # Save checkpoint include: epoch, model, optimizer, loss
  checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),  # Model weights
    'optimizer_state_dict': optimizer.state_dict(),
  }
  torch.save(checkpoint, file_path) 
  print(f"[JV] Checkpoint saved to {file_path}")

def load_checkpoint(model_dir, model, optimizer, epoch=None, is_the_best=False):
  file_path = ""
  if is_the_best:
    file_path = os.path.join(config.CHECKPOINT_DIR, f"{model_dir}/best_checkpoint.tar")
  else:
    if epoch == None:
      cp_files = [f for f in os.listdir(os.path.join(config.CHECKPOINT_DIR, model_dir)) if f.startswith("checkpoint_") and f.endswith(".tar")]
      latest_epoch = 0
      for file in cp_files:
        ep = int(file.split('_')[1].split('.')[0]) # Get epoch num
        if ep > latest_epoch:
            latest_epoch = ep
      epoch = latest_epoch
    file_path = os.path.join(config.CHECKPOINT_DIR, f"{model_dir}/checkpoint_{epoch}.tar")
  print(f"Loading checkpoint from {file_path}")
  checkpoint = torch.load(file_path, map_location=device)
  # Load model and optimizer weights
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return {'epoch': checkpoint['epoch'], 'model': model, 'optimizer': optimizer}