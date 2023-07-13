
import torch
import torch.nn as nn
from dataset import *
from utils import *
import math
import torch.optim as optim
import torch.nn.functional as F
from classifiers.classifier import *
from train import *
from DDPM import *

#check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def schedulerLR(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 15):
    r"""
    This method is used to schedule the learning rate of the reconstruction layer

    Parameters
    ----------
    - cur_lr: current learning rate
    - decay_rate: learning decay rate. defaul = 0.1
    - global_step: default = 1
    - rec_iter: number of recontruction iteration
    
    Returns
    ----------
    - lr: New learning rate

    """

    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.8)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr



def findBestReconstruction(x):
  """
  Method that choose the best reconstruction over all the batch of reconstruction

  Parameters
  ----------
  - x: batch of recontruction loss 
  
  Returns
  ----------
  - The most similar image
  """

  y = torch.Tensor(size=[x.shape[0]])

  for i in range(x.shape[0]):
    # sum over all the pixel reconstruction loss
    y[i] = (x[i].sum().item())

  # choose the most accurate one
  return torch.argmin(y).item()


def reconstruction_module(model, data, lr=25, rec_iter = 4, rand_initi = 5):
    r"""
    Reconstruction module that given a perturbated images it reconstruct
    
    Parameters
    ----------
    - model: diffusion model used to generate the image from noise
    - data: adversarial image
    - lr: learning rate for gradient descent (default = 25)
    - rec_iter: reconstruction iteration/gradient descent iteration (default = 4)
    - rand_initi: Embedding set size (default = 5)
    
    Returns
    ----------

    """

    # prende dal config?
    c, h, w = 1, 28, 28 # TODO

    #this loss is the main one, that used to find 
    loss_fn = nn.MSELoss()

    #this loss is used to take the best reconstruction from all the batch
    loss_fn_ = nn.MSELoss(reduction='none') 

    #Creating a batch of the same image
    data = data[None,:,:,:].repeat(rand_initi,1,1,1).to(device)

    #generate rand_init random noise
    z_hat = torch.randn(size=[rand_initi,c,h,w]).to(device)
    z_hat = z_hat.requires_grad_()

    cur_lr = lr
    
    optimizer = optim.Adam([z_hat], lr=0.02, betas=(0.5, 0.999))
    
    for _ in range(rec_iter):
      
      optimizer.zero_grad()
      
      # Diffusion step
      fake_images = sample(model,z_hat.shape[0],z_hat)

      recon_loss = loss_fn_(fake_images, data)                                                          
      reconstruct_loss = loss_fn(fake_images, data)
            
      reconstruct_loss.backward()            
      optimizer.step()

      # for debug purpose
      #print("Loss:", reconstruct_loss.item(), " LR", optimizer.param_groups[0]['lr'])     
    
      cur_lr = schedulerLR(optimizer, cur_lr, rec_iter=rec_iter)

    z_recons = z_hat.cpu().detach().clone()
    z_gen = fake_images.cpu().detach().clone()

    return  z_gen, recon_loss, z_recons , reconstruct_loss.item()


def reconstruction_pipeline(advdataset, diffusionModel,reciter = 4, randiniti = 5):
   
    c, h, w = 1 , 28, 28 # TODO
    
    bestReconstructions = torch.Tensor(size=[advdataset.shape[0],c,h,w])
    z_ = torch.Tensor(size=[advdataset.shape[0],c,h,w])
    
    # Used to create the ROC curve
    recon_error = torch.Tensor(size=[advdataset.shape[0],1]) 
    
    for i in tqdm(range(advdataset.shape[0])):
         
         x , recon, z_hat, recon_error[i]= reconstruction_module(diffusionModel,advdataset[i], rand_initi=randiniti, rec_iter=reciter)

         best_reconstruced_img = findBestReconstruction(recon)
         bestReconstructions[i] = x[best_reconstruced_img]
         z_[i] = z_hat[best_reconstruced_img]

    # TODO [ ROC CURVE ]
    #torch.save(recon_error, 'name.pt')

    return bestReconstructions, z_
