import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import yaml

def plot_images(images, titolo="test"):
    plt.figure(figsize=(28, 28))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    #plt.show()
    plt.savefig("./imgs/imgrecon/"+str(titolo)+"_img.png")
    plt.close()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def make_grid(grid_images, i):
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(5, 10),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, grid_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.savefig('./imgs/diff/epoch_'+str(i)+'_img.png')
    
def show_images_(images, title="", rows = 2, cols = 5):
    """Shows the provided images as sub-pictures in a square"""
    images = [im.permute(1,2,0).numpy() for im in images]

    # Defining number of rows and columns
    fig = plt.figure(figsize=(20, 20))
    #rows = int(len(images) ** (1 / 2))
    #cols = round(len(images) / rows)

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
    
    # Showing the figure
    #plt.show()
    plt.savefig('./imgs/imgrecon/prova'+title+'.png')
    plt.close()

def make_grid_(grid_images, title, n_row, n_col):
    fig = plt.figure(figsize=(50., 50.))
    grid = ImageGrid(fig, 111, 
                     nrows_ncols=(n_row, n_col),  
                     axes_pad=0.1,  
                     )

    for ax, im in zip(grid, grid_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
        #ax.set_axis_off()
        
    fig.suptitle(title, fontsize=30)
    plt.savefig('./imgs/imgrecon/'+title+'_img.png')
    plt.close()
    #plt.show()

def loadconfigYaml(path):
    with open(path, 'r') as stream:
        config_vars = yaml.safe_load(stream)
    
    return config_vars