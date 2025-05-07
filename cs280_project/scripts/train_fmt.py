# train_fmt.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

model_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
model_path_fmt = os.path.abspath(os.path.join(os.getcwd(), '..', 'FMT'))
print(model_path)
sys.path.append(model_path)
sys.path.append(model_path_fmt)
from FMT.model import AutoregressiveFMT
from FMT.config.AutoregressiveFMTConfig import AutoregressiveFMTConfig
from dataset import QuickDrawDataset
import argparse
from FMT.config.TaskRegistry import find_task_config
from FMT.utils import *

def train_fm_fmt(
    model: AutoregressiveFMT,
    data_path: str,
    epochs: int,
    batch: int,
    fm_steps: int,
    lr: float,
    save_path: str,
    save_freq: int,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- data loaders ---
    train_ds  = QuickDrawDataset(data_path, split="train", num_classes=model.config.general.num_class)
    # val_ds    = QuickDrawDataset(data_path, split="valid")

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=4, pin_memory=(device.type=="cuda")
    )
    # val_loader = DataLoader(
    #     val_ds, batch_size=8, shuffle=False,
    #     num_workers=2, pin_memory=(device.type=="cuda")
    # )
    os.makedirs(save_path, exist_ok=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_losses = []
    # best_loss    = float("inf")
    # best_state   = None

    chunk_size = model.config.FMTConfig.chunk_frames
    prev_chunk_size = model.config.FMTConfig.prev_frames
    for ep in range(epochs):
        model.train()

        for k in range(fm_steps):                                    # ← FM step is outer‑most
            step_loss  = 0.0
            batch_cnt  = 0

            for videos, labels in tqdm(
                    train_loader,
                    desc=f"E{ep+1}/{epochs}  FM‑step {k+1}/{fm_steps}",
                    leave=False):

                # ---- move data to device ----
                videos  = videos.to(device)          # (B, T, H, W)
                labels  = labels.to(device)   # keep first 2 classes
                #print(labels)

                # add channel dim  → (B, T, 1, H, W)
                if videos.ndim == 4:
                    videos = videos.unsqueeze(2)

                B, T, _, H, W = videos.shape
                num_chunks    = T // chunk_size
                if num_chunks == 0:
                    continue

                # ---- iterate over current‑chunk / prev‑chunk pairs ----
                for i in range(num_chunks):
                    start_cur = i * chunk_size
                    end_cur   = start_cur + chunk_size
                    x_cur     = videos[:, start_cur:end_cur]         # (B, chunk_size, 1, H, W)

                    if i == 0:                                      # first chunk has no history
                        x_prev = None
                    else:
                        start_prev = max(0, start_cur - prev_chunk_size)
                        x_prev     = videos[:, start_prev:start_cur] # (B, prev_chunk_size, 1, H, W)

                    # ---- optimisation step ----
                    optimizer.zero_grad()
                    loss = model(x_cur, labels, x_prev)
                    loss.backward()
                    optimizer.step()

                    step_loss += loss.item()
                    batch_cnt += 1

            # ---- logging for this FM‑step ----
            if batch_cnt:
                avg_step_loss = step_loss / batch_cnt
                print(f"[Epoch {ep+1}/{epochs}] FM‑step {k+1}/{fm_steps}  "
                    f"avg‑loss {avg_step_loss:.5f}")

        # ---- end‑of‑epoch bookkeeping ----
        if (ep + 1) % save_freq == 0:
            ckpt_path = os.path.join(save_path, f"model_epoch{ep+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[checkpoint] saved {ckpt_path}")
    """ for ep in range(epochs):
        model.train()
        epoch_loss = 0.0
        batch_cnt  = 0

        for videos, labels in tqdm(train_loader, desc=f"Train Epoch {ep+1}/{epochs}", leave=False):
            # videos: (B, T, H, W), labels: (B, num_class)
            videos = videos.to(device)
            labels = labels.to(device) 
            #switch from 10 classes to 2 classes
            labels = labels[:, :2]
            #print(labels)
            # add channel dim → (B, T, 1, H, W)
            if videos.ndim == 4:
                videos = videos.unsqueeze(2)

            B, T, _, H, W = videos.shape
            num_chunks = T // chunk_size
            if num_chunks == 0:
                continue

            # iterate through consecutive chunk pairs
            for i in range(num_chunks):
                start_cur = i * chunk_size
                end_cur   = start_cur + chunk_size
                x_cur     = videos[:, start_cur:end_cur]
                #x_prev = videos[:, i * chunk_size : (i + 1) * chunk_size]    # (B, T_cur, 1, H, W)
                #x_cur  = videos[:, (i + 1) * chunk_size : (i + 2) * chunk_size]
                if i == 0:
                    x_prev = None
                else:
                    # you can choose how to align your prev window:
                    # e.g. right before the current chunk, or a fixed-size lookback:
                    start_prev = max(0, start_cur - prev_chunk_size)
                    end_prev   = start_cur
                    x_prev     = videos[:, start_prev:end_prev]
                
                for k in range(fm_steps):
                    optimizer.zero_grad()
                    loss = model(x_cur, labels, x_prev)
                    loss.backward()
                    #print(get_grad_norm(model))

                    #if gradient_clip > 0:
                        #torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                    optimizer.step()

                    epoch_loss  += loss.item()
                    train_losses.append(loss.item())
                    print(f"[E{ep+1}/{epochs} chunk {i+1}/{num_chunks} S{k+1}/{fm_steps}] train loss {loss.item():.5f}")

                batch_cnt += 1

        # end of epoch
        avg_loss = epoch_loss / (max(batch_cnt, 1) * fm_steps)
        print(f"Epoch {ep+1}/{epochs} : train loss {avg_loss:.5f}")

        if batch_cnt:
            #scheduler.step()
            pass

        # checkpoint
        if (ep + 1) % save_freq == 0:
            path = os.path.join(save_path, f"model_epoch{ep+1}.pth")
            torch.save(model.state_dict(), path)
            print(f"[checkpoint] saved {path}") """

    # # save best
    # if best_state is not None:
    #     best_path = os.path.join(save_path, "best_model.pth")
    #     torch.save(best_state, best_path)
    #     print(f"[checkpoint] saved best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/quickdraw_50_apple_star.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch',  type=int, default=512)
    parser.add_argument('--lr',  type=float, default=1e-3)
    parser.add_argument('--fm_steps',     type=int, default=100)
    parser.add_argument('--save_freq',     type=int, default=5)
    parser.add_argument('--save_path',   default="./checkpoints/autoregressive_fmt_50")
    parser.add_argument('--task', type=str, default='fmt')
    parser.add_argument('--pretrained_path', type=str, default='None')

    args = parser.parse_args()
    # --- training ---

    cfg = find_task_config(args.task)()
    model = AutoregressiveFMT(cfg)


    train_fm_fmt(
        model=model,
        data_path=args.data_path,
        epochs=args.epochs,
        batch=args.batch,
        fm_steps=args.fm_steps,
        lr=args.lr,
        save_path=args.save_path,
        save_freq=args.save_freq,
    )
