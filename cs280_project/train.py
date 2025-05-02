import torch
from Unet import *
from typing import *
from train_utils import *
import matplotlib as plt
from tqdm import tqdm
from dataset import QuickDrawDataset
from GRU import HistoryEncoder
import os

def train_fm_cond(fm_model, epoch, lr, train_dataloader, val_dataloader, device, in_channel, gradient_clip = 0, save_path = './checkpoints'):
  optimizer = torch.optim.Adam(fm_model.parameters(), lr=lr) #train the whole model
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.89)
  model = fm_model.to(device)
  train_loss = []
  chunk_size = 12
  best_loss = float('inf')
  best_model = None
  save_freq = 10
  validating = False
  
  for ep in range(epoch):
    print(f"Epoch {ep+1}/{epoch}")
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    #video shape: (batch_size, frames, H, W)
    for video, labels in tqdm(train_dataloader):
      video = video.to(device)
      labels = labels.to(device)
      batch_size = video.size(0)
      
      num_frames = video.size(1)
      num_chunks = num_frames // chunk_size
      
      # 只处理有足够块数的视频
      if num_chunks < 2:
        continue
      
      # 初始化GRU隐藏状态
      h_state = model.GRU.init_state(batch=batch_size, device=device)
      
      # 按时间顺序处理每个块
      for i in range(num_chunks-1):  # 最后一个块作为目标，不作为输入
        # 获取当前块作为输入
        current_chunk = video[:, i*chunk_size:(i+1)*chunk_size]
        # 获取下一个块作为目标
        next_chunk = video[:, (i+1)*chunk_size:(i+2)*chunk_size]
        
        # 计算损失，model.forward会自动处理GRU
        loss, h_state = model.forward(next_chunk, current_chunk, h_state, labels)
        
        # 记录和反向传播
        train_loss.append(loss.item())
        epoch_loss += loss.item()
        batch_count += 1
        
        loss.backward()
        

        if gradient_clip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()
        optimizer.zero_grad()

        #detach the h_state to avoid double backprop
        h_state = h_state.detach()
    
    # 打印本轮训练的平均损失
    avg_epoch_loss = epoch_loss / max(batch_count, 1)
    print(f"Epoch {ep+1} average loss: {avg_epoch_loss:.6f}")
    
    # 验证和可视化
    if validating:
      print("Validating...")
      val_loss = validate_and_visualize(model, val_dataloader, in_channel, 
                                     device=device,
                                     autoregressive_steps=0, 
                                     visualization=(ep % 5 == 0))
      print(f"Validation loss: {val_loss:.6f}")
    
      if val_loss < best_loss:
        best_loss = val_loss
        best_model = model.state_dict()
    
    if (ep + 1) % save_freq == 0:
      torch.save(model.state_dict(), os.path.join(save_path, f'model_{ep+1}.pth'))
    
    scheduler.step()
  
  if best_model:
    torch.save(best_model, os.path.join(save_path, 'best_model.pth'))

  # 绘制训练损失曲线
  import matplotlib.pyplot as plt
  plt.figure(figsize=(10, 6))
  x_values = list(range(len(train_loss)))
  plt.plot(x_values, train_loss)
  plt.title('Training Loss')
  plt.xlabel('Iteration')
  plt.ylabel('MSE loss')
  plt.yscale('log')
  plt.savefig('training_loss.png')
  plt.close()
    
### Flow matching training
# def train_fm_cond(fm_model, epoch, lr, train_dataloader, val_dataloader, device, in_channel, gradient_clip = 0):
#   optimizer = torch.optim.Adam(fm_model.unet.parameters(), lr=lr)
#   scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.89)
#   model = fm_model.to(device)
#   train_loss = []

#   num_chunks = 60 // 12

#   for ep in range(epoch):
#     model.train()
#     # video = [bs, frames=60, 1, 256, 256]
#     for video, c in tqdm(train_dataloader):
#       for i in range(num_chunks):
#         # video = [bs, frames=60, 1, 256, 256]
#         image = video[:, i * 12: (i + 1) * 12, :, :, :]
#         if i == 0:
#           x_p = torch.zeros_like(image)
#           h_next = model.GRU.init_state(batch=x_p.size(0), device=device)
#         else:
#           x_p = video[:, (i - 1) * 12: i * 12, :, :]
#         h_vec, h_next = model.GRU(x_p, h_next)

#       if torch.all(x_p == 0):
#         h_vec = model.GRU.init_state(batch=x_p.size(0), device=device)
#       h_vec, h_next = model.GRU(x_p, h_vec)

#       loss = model.forward(image.to(device), x_p.to(device), h_vec, c.to(device))
#       train_loss.append(loss.item())
#       loss.backward()
#       if gradient_clip:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
#       optimizer.step()
#       optimizer.zero_grad()

#     validate_and_visualize(model, val_dataloader, in_channel, autoregressive_steps=0)

#     scheduler.step()

#   x_values = list(range(len(train_loss)))
#   plt.plot(x_values, train_loss)
#   plt.title('Training Loss')
#   plt.xlabel('Iteration')
#   plt.ylabel('MSE loss')
#   plt.yscale('log')



## TODO: this function is currently buggy
## TODO (1) use `autoregressive_sample` to generate the video, then compute MSE on val data, use the first 12 frames as x_p
def validate_and_visualize(model, val_dataloader, in_channel, autoregressive_steps = 0, visualization = False):
    losses = []
    loss_f = torch.nn.MSELoss()
    chunk_size = 12

    for image, x_p, c in tqdm(val_dataloader):
      out = model.sample(c, x_p, (x_p.size(-2), x_p.size(-1)))
      loss = loss_f(out, image)
      losses.append(loss)

      ### Visualization 
      if visualization:
        fig, axs = plt.subplots(1, x_p.size(0) * 2, figsize=(50, 20))
        noisy_images = []
        ground_truth = []
        for i in range(x_p.size(0)):
            noisy_images.append(out[i])
            ground_truth.append(image[i])
        i = 0
        show_truth = False
        for _, ax in enumerate(axs):
            if show_truth:
                ax.imshow(ground_truth[i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
                show_truth = False
                i += 1
            else:
                ax.imshow(noisy_images[i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
                show_truth = True
            ax.axis('off')  
        plt.tight_layout()
        plt.show()
      ###

      for _ in range(autoregressive_steps):
        x_p = torch.cat((x_p, out), dim = 1)[:, -in_channel * chunk_size: , : , :]
        out = model.sample(c, x_p, (x_p.size(-2), x_p.size(-1)))

        ### Visualization 
        if visualization:
            fig, axs = plt.subplots(1, x_p.size(0) * 2, figsize=(50, 20))
            noisy_images = []
            ground_truth = []
            for i in range(x_p.size(0)):
                noisy_images.append(out[i])
                ground_truth.append(image[i])
            i = 0
            show_truth = False
            for _, ax in enumerate(axs):
                if show_truth:
                    ax.imshow(ground_truth[i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
                    show_truth = False
                    i += 1
                else:
                    ax.imshow(noisy_images[i].detach().cpu().numpy(), cmap='gray', interpolation='nearest')
                    show_truth = True
                ax.axis('off')  
            plt.tight_layout()
            plt.show()
        ### 

        

      
if __name__ == "__main__":
    model = FlowMatching(UNet(1, 64, 64), GRU=HistoryEncoder(in_frames=12))
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 数据路径
    data_path = 'data/quickdraw_1k.npz'
    
    # 创建数据加载器
    batch_size = 8
    predict_frames = 1
    
    # 创建数据集
    train_dataset = QuickDrawDataset(data_path, split='train')
    val_dataset = QuickDrawDataset(data_path, split='valid')
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,  # 在这里设置批处理大小
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,  # 同上
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 开始训练
    train_fm_cond(model, 20, 1e-2, train_loader, val_loader, device, 1)