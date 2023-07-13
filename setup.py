import os

"""
    File that creats the folders  neccessary for the project
    ./img to save images
    ./pretrained to save pretrained models
    ./re_roc to save the reconstruction error to create roc curve plot
    ./re_roc to save the roc curve plot
"""



path = [ './pretrained', './pretrained/diffusion', './pretrained/diffusion/MNIST',
         './pretrained/diffusion/KMNIST', './pretrained/MNIST', './pretrained/KMNIST',
          './re_roc','./re_roc/AA', './re_roc/DF' , './re_roc/EN', './re_roc/EOTPGD', 
          './re_roc/FGSM', './re_roc/PGD', './re_roc/SA', './re_roc/roc_plot']


for p in path:
    
    if not os.path.exists(p):
        os.mkdir(p)
        print("Folder %s created!" % p)
    else:
        print("Folder %s already exists" % p)

