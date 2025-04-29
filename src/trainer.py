import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src import helper

class Trainer:
  def __init__(self, model, model_dir_name, train_loader, val_loader, optimizer, log_dir):
    self.model = model
    self.model_dir_name = model_dir_name
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.tensorboard_writer = SummaryWriter(log_dir=log_dir)

  def run_epoch(self, epoch, is_val_mode=False):
    self.model.train() if not is_val_mode else self.model.eval() # Put the model into training mode or valuating mode
    total_loss = 0
    dataloader = self.train_loader if not is_val_mode else self.val_loader
    context = torch.no_grad() if is_val_mode else torch.enable_grad()
    with context:
      for batch in tqdm(dataloader, desc=f"{'Validation' if is_val_mode else 'Training'} Epoch {epoch}"):
        input_ids = batch['input_ids'].to(helper.device())
        attention_mask = batch['attention_mask'].to(helper.device())
        labels = batch['labels'].to(helper.device())
        # Forward pass
        outputs = self.model(
          input_ids=input_ids, 
          attention_mask=attention_mask,
          labels=labels
        )
        loss = outputs.loss # Get loss value
        if not is_val_mode:
          # Backward pass
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

      total_loss += loss.item()
      # if index%5 == 0: print(f"[JV] Batch {index}, Loss: {loss.item()}")

    average_loss = total_loss/len(dataloader)
    print(f"[JV] Epoch {epoch}, {'Validation' if is_val_mode else 'Training'} Loss: {average_loss}")
    print('='*40)
    self.tensorboard_writer.add_scalar(f"Loss/{'Validation' if is_val_mode else 'Training'}", average_loss, epoch)
    return average_loss

  def train(self, n_epochs):
    print(f"{'='*20} TRAIN MODEL {'='*20}")
    best_loss = float("inf")
    best_at = -1
    for epoch in range(n_epochs):
      train_loss = self.run_epoch(epoch)
      val_loss = self.run_epoch(epoch, is_val_mode=True)
      # Save checkpoint
      if val_loss < best_loss:
        best_loss = val_loss
        best_at = epoch
        helper.save_checkpoint(model_dir=self.model_dir_name, epoch=epoch, model=self.model, optimizer=self.optimizer, is_the_best=True)
      helper.save_checkpoint(model_dir=self.model_dir_name, epoch=epoch, model=self.model, optimizer=self.optimizer)
      if epoch != 0 and epoch % 3 == 0: helper.save_best_checkpoint_to_gdrive(model_dir=self.model_dir_name)
    if n_epochs % 3 != 0: helper.save_best_checkpoint_to_gdrive(model_dir=self.model_dir_name)
    print(f"[JV] Training completed. Best Validation Loss: {best_loss} | Epoch: {best_at}")