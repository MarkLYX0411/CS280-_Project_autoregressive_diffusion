import torch
from Unet import *
from typing import *
from train_utils import *
import matplotlib as plt
from tqdm import tqdm
from dataset import QuickDrawDataset
from GRU import HistoryEncoder
import os

def train_fm_cond(
    fm_model,
    epoch,
    fm_epoch,
    lr,
    train_dataloader,
    val_dataloader,
    device,
    in_channel,
    gradient_clip=0.0,
    save_path="./checkpoints_0505_64_resolution_32_unet_1k",
    chunk_size=12,
    save_freq=5,
    validating=False,
):
  os.makedirs(save_path, exist_ok=True)

  model = fm_model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.89)

  train_losses, best_loss = [], float("inf")
  best_state = None

  # get # of trainable parameters
  total_params = 0
  for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
  print(f"Total Trainable Params: {total_params}")

  for ep in range(epoch):
    model.train()
    epoch_loss, batch_cnt = 0.0, 0

    for video, labels in tqdm(train_dataloader, leave=False):
        video, labels = video.to(device), labels.to(device)
        B, T = video.shape[:2]
        num_chunks = T // chunk_size
        if num_chunks < 2:
            continue

        h_state = model.GRU.init_state(batch=B, device=device)

        # iterate through consecutive block pairs
        for i in range(num_chunks - 1):
            x_chunk = video[:, i * chunk_size : (i + 1) * chunk_size]
            y_chunk = video[:, (i + 1) * chunk_size : (i + 2) * chunk_size]

            for k in range(fm_epoch):
                #retain = k < fm_epoch - 1
                optimizer.zero_grad(set_to_none=True)
                h_vec, h_curr = model.get_GRU_feature(x_chunk, h_state)

                loss = model.forward_fixed_GRU(y_chunk, h_vec, labels)
                loss.backward()

                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                optimizer.step()
                epoch_loss += loss.item()
                train_losses.append(loss.item())

                print(f"Epoch {ep+1}/{epoch} fm_epoch {k+1}/{fm_epoch} | train loss {loss.item():.5f}")

            batch_cnt += 1
            h_state = h_curr.detach()

        # ----- epoch end -----
        mean_loss = epoch_loss / (max(batch_cnt, 1) * fm_epoch)
        print(f"Epoch {ep+1}/{epoch} | train loss {mean_loss:.5f}")

        if batch_cnt:            # avoid decay if nothing was trained
            scheduler.step()

        # ----- validation -----
        if validating:
            val_loss = validate_and_visualize(
                model, val_dataloader, in_channel,
                device=device, autoregressive_steps=0,
                visualization=(ep % 5 == 0),
            )
            print(f"  val loss {val_loss:.5f}")
            if val_loss < best_loss:
                best_loss, best_state = val_loss, {
                k: v.cpu() for k, v in model.state_dict().items()
                }

        if (ep + 1) % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{ep+1}.pth"))

    if best_state is not None:
        torch.save(best_state, os.path.join(save_path, "best_model.pth"))

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
    model = FlowMatching(UNet(1, 32, 32), GRU=HistoryEncoder(in_frames=12))
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 数据路径
    data_path = '/home/yujunwei/CS280-_Project_autoregressive_diffusuon/quickdraw_1k5_apple_cat.npz'
    
    # 创建数据加载器
    batch_size = 8
    predict_frames = 1
    
    # 创建数据集
    train_dataset = QuickDrawDataset(data_path, split='train')
    val_dataset = QuickDrawDataset(data_path, split='valid')
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,  # 在这里设置批处理大小
        shuffle=True,
        num_workers=10,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # 同上
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 开始训练
    train_fm_cond(model, 100, 100, 1e-3, train_loader, val_loader, device, 1)