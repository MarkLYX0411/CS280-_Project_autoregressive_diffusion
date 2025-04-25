import torch
from model import *
from typing import *
from train_utils import *
import matplotlib as plt
import tqdm
    
### Flow matching training
def train_fm_cond(fm_model, epoch, lr, train_dataloader, val_dataloader, device, in_channel, gradient_clip = 0):
  optimizer = torch.optim.Adam(fm_model.unet.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.89)
  model = fm_model.to(device)
  train_loss = []

  for ep in range(epoch):
    model.train()
    for image, x_p, c in tqdm(train_dataloader):
      loss = model.forward(image.to(device), x_p.to(device), c.to(device), in_channel = in_channel)
      train_loss.append(loss.item())
      loss.backward()
      if gradient_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
      optimizer.step()
      optimizer.zero_grad()

    validate_and_visualize(model, val_dataloader, in_channel, autoregressive_steps=0)

    scheduler.step()

  x_values = list(range(len(train_loss)))
  plt.plot(x_values, train_loss)
  plt.title('Training Loss')
  plt.xlabel('Iteration')
  plt.ylabel('MSE loss')
  plt.yscale('log')


def validate_and_visualize(model, val_dataloader, in_channel, autoregressive_steps = 0, visualization = False):
    losses = []
    loss_f = torch.nn.MSELoss()

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
        x_p = torch.cat((x_p, out), dim = 1)[:, in_channel: , : , :]
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
    model = FlowMatching(UNet(1, 64, 64, 10))
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    ###  TODO
    train_loader = ...
    val_loader = ...
    ###
    train_fm_cond(model, 20, 1e-2, train_loader, val_loader, device, 1)