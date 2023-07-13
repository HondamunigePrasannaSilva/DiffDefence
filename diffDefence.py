from attackpipeline import *
import reconstructphase as rp
from classifiers.classifier import *
from train import *

from unet.unet import *
from DDPM import *

import torch
from dataset import * 
from utils import *
import time
import os

import argparse

"""
    Main file
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = {
    'dataset':              'MNIST',        # MNIST or KMNSIT
    'classifier':           'classifier_a', # name of the pretrained classifier
    'sub_classifier':       None,           # clasifier used to create adv example for black box attack
    'typeA':                None,           # type of the main classifier [ classifier_a or classifier_b]
    'typeB':                None,           # type of the sub classifier [ classifier_a or classifier_b]
    'diffusion_name':       'MNIST_ty_100', # Diffusion model name used to save the parameters
    'diffusion_step':       15,             # Diffusion step to generate the image
    'batch_size':           32,
    'test_size':            512,            # Number of images to attack!
    'reconstruction_iter':  4,              # Iteration of the reconstruction phase
    'reconstruction_init':  5,              # Num. of restart of the reconstruction phase

    # Make sure that these are the same of the trained model!
    'beta_start':           0.0001,
    'beta_end':             0.02,
}
def main():
        
    parser = argparse.ArgumentParser(description='DiffDefence: main module!')
    
    parser.add_argument("--dataset",             type=str, default="MNIST", help='dataset (MNIST-KMNIST)')
    parser.add_argument("--classifier",          type=str, default="classifier_a", help='Name of the main classifier')
    parser.add_argument("--typeA",               type=str, default="classifier_a", help='classifier_a or classifier_b')
    parser.add_argument("--sub_classifier",      type=str, default=None, help='classifier_a or classifier_b [Black box]')
    parser.add_argument("--typeB",               type=str, default="classifier_b", help='classifier_a or classifier_b')
    parser.add_argument("--diffusion_name",      type=str, default="MNIST_ty_100", help='Diffusion model name')
    parser.add_argument("--diffusion_step",      type=int, default=15, help='diffusion step to generate the image')
    parser.add_argument("--batch_size",          type=int, default="32", help='Batch size')
    parser.add_argument("--test_size",            type=int, default=64, help='number of image to attack, must be less than testset of dataset')
    parser.add_argument("--reconstruction_iter", type=int, default=4, help='diffdefence reconstuction iteration')
    parser.add_argument("--reconstruction_init", type=int, default=5, help='diffdefence reconstuction initialization')
    parser.add_argument("--beta_start",          type=float, default=0.0001, help='Beta start for diffusion model')
    parser.add_argument("--beta_end",            type=float, default=0.02, help='Beta end for diffusion model')

    args = parser.parse_args()

    # Load the configuration on the dict!
    config['dataset']             = args.dataset
    config['classifier']          = args.classifier    
    config['sub_classifier']      = args.sub_classifier
    config['typeA']               = args.typeA
    config['typeB']               = args.typeB
    config['diffusion_name']      = args.diffusion_name
    config['diffusion_step']      = args.diffusion_step
    config['batch_size']          = args.batch_size
    config['test_size']           = args.test_size
    config['reconstruction_iter'] = args.reconstruction_iter
    config['reconstruction_init'] = args.reconstruction_init
    config['beta_start']          = args.beta_start
    config['beta_end']            = args.beta_end


    attacks = { 
            'FGSM':                FGSM_Attack_CH ,     # FGSM_Attack if you want to use ART version!
            'PGD':                 PGD_Attack,
            'Deep Fool':           DF_Attack,
            'EOTPGD':              EOTPGD_Attack,
            'AutoAttack':          AA_Attack,
            'Square Attack':       SA_Attack,
            'Elastic Net Attack':  EN_Attack
         }

    for attackName in attacks:
        mainPipeline(config, attacks[attackName], attackName)


def getClassifier(config, submodel = False):
    """
        Function to get the pretrained classifier!
    """
    if submodel == False :
        path = "./pretrained/"+config['dataset']+"/"+config['classifier']+".pt"
        model = classifiers[config['typeA']].to(device)    
    
    if submodel == True:
        path = "./pretrained/"+config['dataset']+"/"+config['sub_classifier']+".pt"
        model = classifiers[config['typeB']].to(device)
    
    assert os.path.exists(path) == True, f"Path: {path} do not exists"
    model.load_state_dict(torch.load(path))
    
    return model

def getGenerator(config):
    
    # Get UNET
    network = UNet()
    network = network.to(device)

    # GET Diffusion
    model = DDPM(network, config['diffusion_step'], beta_start=config['beta_start'], beta_end=config['beta_end'])
    model.load_state_dict(torch.load(f"./pretrained/diffusion/{config['dataset']}/{config['diffusion_name']}.pt"))
    
    return model

def mainPipeline(config, attack,attackName):
    
    torch.manual_seed(0)

    # GET DIFFUSION MODEL
    diffusion = getGenerator(config) 
    
    # GET CLASSIFIERS
    model = getClassifier(config)

    # For black box attack
    if config['sub_classifier'] is not None:
        smodel = getClassifier(config, submodel=True)
    
    # IMAGES TO ATTACK
    image_to_attack = getData(datasetname=config['dataset'], typedata="test", batch_size=config['batch_size'], test_size=config['test_size'])
    
    # TRASFORM IT IN ADVERSARIAL IMAGES - It returns adversarial images and the original labels
    if config['sub_classifier'] is not None:    
        # If Blackbox setting, use subtitute model to create the adversarial images
        adv_images, labels = attack(smodel,config['dataset'],config['classifier'], image_to_attack, config['batch_size'], config['typeA'])
    else:
        adv_images, labels = attack(model,config['dataset'],config['classifier'],image_to_attack,config['batch_size'], config['typeB'])
      
    #RECONSTRUCTION PHASE
    start_time = time.time()
    reconstructImages, z = rp.reconstruction_pipeline(advdataset=adv_images, diffusionModel=diffusion, reciter = config['reconstruction_iter'], randiniti = config['reconstruction_init'])
    finish_time = time.time()-start_time

    

    print(f"---------------------{attackName}----------------------------------------")    
    #print("ACCURACY ON ORIGINAL IMAGES:", testProva(model, d, l, n) )
    print(f"ACCURACY ON ADVERSARIAL IMAGES:{testadv(model, adv_images, labels, config['test_size']):.2f}%" )
    print(f"ACCURACY ON RECONSTRUCTED IMAGES:{testadv(model, reconstructImages, labels, config['test_size']):.2f}%")
    print(f"Time to reconstruct one image {finish_time/config['test_size']:.4f}s")
    print(f"-------------------------------------------------------------------------")



    # Print first 10 images!
    make_grid_(adv_images.detach().squeeze().cpu(),f"Adversaial_{config['dataset']}_{attackName}", 2, 5)
    make_grid_(next(iter(image_to_attack))[0].detach().cpu().squeeze(),f"Original_{config['dataset']}_{attackName}", 2, 5)
    make_grid_(reconstructImages[0:10].detach().cpu().squeeze(),f"Reconstruct_{config['dataset']}_{attackName}", 2, 5)
    



if __name__ == '__main__':
    
    main()        

