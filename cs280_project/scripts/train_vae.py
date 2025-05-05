import os, math, argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from FMT.vae import VAE
from dataset import QuickDrawDataset    
import torch.nn.functional as F
from torch.utils.data import Dataset
import bisect

class FrameDataset(Dataset):
    """
    Flattens the QuickDraw *video* dataset into individual *frames*.

    Each __getitem__ returns:
        frame : torch.FloatTensor  (1, H, W)  — pixel range [0,1]
    """
    def __init__(self, npz_path: str, split: str = "train"):
        super().__init__()
        self.videos = QuickDrawDataset(npz_path, split=split)

        # build cumulative frame counts so we can binary‑search
        self.cum_frames = []
        running = 0
        for video, _ in self.videos:          # video : (T, H, W) numpy/torch
            running += video.shape[0]
            self.cum_frames.append(running)   # last entry = total_frames
        self.total_frames = running

    # -------- Dataset API --------
    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_frames:
            raise IndexError(idx)

        # locate which video this frame lives in
        vid_idx = bisect.bisect_right(self.cum_frames, idx)
        first_frame_of_video = 0 if vid_idx == 0 else self.cum_frames[vid_idx - 1]
        frame_idx = idx - first_frame_of_video

        video, _ = self.videos[vid_idx]       # video: (T, H, W) tensor
        frame = video[frame_idx]              # (H, W)

        # ensure tensor float32 in [0,1] and add channel dim
        if not torch.is_tensor(frame):
            frame = torch.from_numpy(frame)
        frame = frame.float()
        if frame.max() > 1.0:
            frame = frame / 255.0
        return frame.unsqueeze(0)

def train_vae(
        data_path:   str   = 'data/compressed_quickdraw_1k.npz',
        batch_size:  int   = 256,
        epochs:      int   = 60,
        lr:          float = 1e-3,
        kl_beta:     float = 1e-2,
        save_dir:    str   = './checkpoints/vae',
        save_freq:   int   = 5,
        num_workers: int   = 4,
        amp:         bool  = True
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vae = VAE(z_ch=64, kl_beta=kl_beta).to(device)
    opt  = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.9,0.99))
    sched= torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)

    train_ds = FrameDataset(data_path, split='train')
    val_ds   = FrameDataset(data_path, split='valid')

    train_ld = DataLoader(train_ds, batch_size, True,  num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size, False, num_workers=num_workers, pin_memory=True)

    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device == 'cuda'))

    best_val = math.inf
    for ep in range(1, epochs+1):

        # ----- training -----
        vae.train()
        running_l1, running_kl = 0.0, 0.0
        for x in tqdm(train_ld, desc=f'Epoch {ep}/{epochs}'):
            x = x.to(device)
            with torch.amp.autocast('cuda', enabled=(amp and device == 'cuda')):
                out = vae(x)
                loss = out['loss']
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            running_l1 += out['recon_l1'].item() * x.size(0)
            running_kl += out['kl'].item()       * x.size(0)

        sched.step()
        train_l1 = running_l1 / len(train_ds)
        train_kl = running_kl / len(train_ds)
        print(f'  train L1 {train_l1:.4f}  KL {train_kl:.4f}')

        # ----- validation -----
        vae.eval()
        val_l1, val_kl = 0.0, 0.0
        with torch.no_grad():
            for x in val_ld:
                x = x.to(device)
                out = vae(x)
                val_l1 += out['recon_l1'].item() * x.size(0)   # keep per‑pixel mean
                val_kl += out['kl'].item()       * x.size(0)

        val_l1 /= len(val_ds)
        val_kl /= len(val_ds)
        print(f'  val   L1 {val_l1:.4f}  KL {val_kl:.4f}')

        # ----- checkpointing -----
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(vae.state_dict(), os.path.join(save_dir, 'best.pth'))
            print('  ↳ new best model saved.')
        if ep % save_freq == 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f'vae_{ep:03d}.pth'))

    print(f'Done. Best val L1 = {best_val:.4f}')

# ----------  CLI entry  ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/compressed_quickdraw_1k.npz')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch',  type=int, default=256)
    parser.add_argument('--lr',     type=float, default=1e-3)
    parser.add_argument('--kl',     type=float, default=1e-2)
    parser.add_argument('--save',   default='./checkpoints/vae')
    args = parser.parse_args()

    train_vae(
        data_path   = args.data,
        batch_size  = args.batch,
        epochs      = args.epochs,
        lr          = args.lr,
        kl_beta     = args.kl,
        save_dir    = args.save,
    )