import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import *
from unet.unet import *
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparamters for the training pipeline
config = {
        'LR': 1e-3,
        'EPOCHS': 500,
        'BATCH_SIZE': 128,
        'N_TIMESTEPS':100,
        'BSTART':0.00001,
        'BEND':0.02,
        'DATASET': 'MNIST',
        'MODEL_TITLE':'name',
}

def main():
    parser = argparse.ArgumentParser(description='DiffDefence: Train Diffusion module')
    
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--dataset", type=str, default="MNIST", help='dataset (MNIST-KMNIST)')
    parser.add_argument("--epochs", type=int, default="501", help='Training epochs')
    parser.add_argument("--batch_size", type=int, default="1024", help='Batch size')
    parser.add_argument("--bstart", type=float, default=0.00001, help='Beta start')
    parser.add_argument("--bend", type=float, default=0.02, help='Beta end')
    parser.add_argument("--model_title", type=str, default="modelname", help='Model name')
    parser.add_argument("--n_timestep", type=int, default="100", help='diffusion time step')

    args = parser.parse_args()

    config["LR"] = args.lr
    config["DATASET"] = args.dataset
    config["EPOCHS"] = args.epochs
    config["BATCH_SIZE"] = args.batch_size
    config["MODEL_TITLE"] = args.model_title
    config["BSTART"] = args.bstart
    config["BEND"] = args.bend
    config["N_TIMESTEPS"] = args.n_timestep


    trainDDPM(config)

def show_images(images, title): # Da spostare dentro util?
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1,2,0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx])
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    
    plt.savefig(f'./imgs/diff/{title}.png')
    plt.close()


class DDPM(nn.Module):
    def __init__(self, network, num_timesteps, beta_start=0.0001, beta_end=0.02) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.network = network
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1,1,1) # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1,1,1,1) # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)
    
    def step(self, model_output, timestep, sample):
        t = timestep
        coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1,1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1,1,1)
        pred_prev_sample = coef_first_t*(sample-coef_eps_t*model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(device)
            variance = ((self.betas[t] ** 0.5) * noise)
            
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def stepDDIM(self, model_output, timestep, sample):
        t = timestep
        variance = 0
        alpha_hat = self.alphas_cumprod[t][:, None, None, None]
        pred_origina_sample = (sample-(1-alpha_hat)**0.5 * model_output)/(alpha_hat**0.5)
        s = sample
        if(t > 0):
            alpha_hat = self.alphas_cumprod[t][:, None, None, None]
            alpha_hat_prev = self.alphas_cumprod[t-1][:, None, None, None]

            beta_prod_t_prev = 1-alpha_hat_prev
            beta_prod_t = 1-alpha_hat

            sigma_t = (beta_prod_t_prev / beta_prod_t) * (1 - (alpha_hat / alpha_hat_prev))
            
            noise = torch.randn_like(model_output).to(device)
            
            
            pred_sample_direction = torch.sqrt(1-alpha_hat_prev-sigma_t)*model_output

            variance = (sigma_t**0.5)*noise 
            
            s = torch.sqrt(alpha_hat_prev)*pred_origina_sample + variance + pred_sample_direction
        else:
            #sample = pred_origina_sample
            s = self.step(model_output, timestep, sample)
        
        return s 
    
    def getAlphaCum(self,t):
        return self.alphas_cumprod[t-1][:,None,None,None]

def training_loop(model, dataloader, optimizer,config):
    """Training loop for DDPM"""

    global_step = 0
    losses = []
    
    for epoch in range(config['EPOCHS']):
        model.train()
        progress_bar = tqdm(total=len(dataloader), leave=False)
        progress_bar.set_description(f"Epoch {epoch}/{config['EPOCHS']}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, config['N_TIMESTEPS'], (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        
        torch.save(model.state_dict(), f"./pretrained/diffusion/{config['DATASET']}/{config['MODEL_TITLE']}.pt")
        if(epoch%10==0):
            generated = generate_image(model, 100, 1, 28)
            show_images(generated, f"Final result ({config['DATASET']})")
        progress_bar.close()

def generate_image(ddpm, sample_size, channel, size):
    """Generate the image from the Gaussian noise"""
    frames = []
    ddpm.eval()
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)
        
        for i, t in enumerate(tqdm(timesteps, leave=False)):
            time_tensor = (torch.ones(sample_size,1) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor) # senza grad
            sample = ddpm.step(residual, time_tensor[0], sample)

        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames

def sample(ddpm, sample_size, z_t):
    
    ddpm.eval()
    timesteps = list(range(ddpm.num_timesteps))[::-1]
    #timesteps = list(range(50))[::-1]

    z_t = z_t.to(device)
    
    for i, t in enumerate(timesteps):
        time_tensor = (torch.ones(sample_size,1) * t).long().to(device)
        
        with torch.no_grad():
            residual = ddpm.reverse(z_t, time_tensor)

        #if(i%2==0):
        #    z_t = ddpm.step(residual, time_tensor[0], z_t)
        #else:
        z_t = ddpm.stepDDIM(residual, time_tensor[0], z_t)

    #z_t = z_t.clamp(-1, 1) 
    z_t = torch.sigmoid(z_t)
    #z_t = torch.tanh(z_t)          
    return z_t

def sample_(ddpm, sample_size, z_t):
    
    ddpm.eval()
    
    timesteps = list(range(ddpm.num_timesteps))[::-1]


    z_t = z_t.to(device)


    e = torch.randn_like(z_t)
    t = (torch.ones(sample_size) * 15).long().to(device)
    
    aT = ddpm.getAlphaCum(t)

    z_t = torch.sqrt(aT)*z_t + torch.sqrt(1-aT)*e
    
    for i, t in enumerate(tqdm(timesteps, leave=False)):
        time_tensor = (torch.ones(sample_size,1) * t).long().to(device)
        
        with torch.no_grad():
            residual = ddpm.reverse(z_t, time_tensor)

        z_t = ddpm.step(residual, time_tensor[0], z_t)


    z_t = torch.tanh(z_t)          
    return z_t

def trainMNIST(): # DELETE
    learning_rate = 1e-3
    num_epochs = 501
    num_timesteps = 100
    network = UNet()
    network = network.to(device)
    model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    dataloader , _ = getData(datasetname="MNIST", batch_size=512, typedata="both")
    training_loop(model, dataloader, optimizer, num_epochs, num_timesteps,title="MNIST_ty_100", device=device) 

def trainKMNIST(): # DELETE
    learning_rate = 1e-3
    num_epochs = 1001
    num_timesteps = 100
    network = UNet()
    network = network.to(device)
    model = DDPM(network, num_timesteps, beta_start=0.00001, beta_end=0.02, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    dataloader , _ = getData(datasetname="KMNIST", batch_size=1024, typedata="both")
    training_loop(model, dataloader, optimizer, num_epochs, num_timesteps,title="KMNIST_100", device=device)

def trainDDPM(config):
    network = UNet()
    network = network.to(device)
    model = DDPM(network, config['N_TIMESTEPS'], beta_start=config['BSTART'], beta_end=config['BEND'])
    optimizer = torch.optim.Adam(network.parameters(), lr=config["LR"])
    dataloader , _ = getData(datasetname=config['DATASET'], batch_size=config['BATCH_SIZE'], typedata="both")
    training_loop(model, dataloader, optimizer,config)







if __name__ == '__main__':

   main()
