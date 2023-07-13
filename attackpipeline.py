import dataset
import utils
from classifiers.classifier import *
from train import *
from torch.utils.data import DataLoader , TensorDataset
from train import *
from dataset import *
import numpy as np
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from torch import optim


#  ------ CLEVER HANS ------
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)

#  ------ TORCH ATTACKS ------
from torchattacks import AutoAttack, EOTPGD

#  ------ ADVERSARIAL ROBUSTNESS TOOLBOX ------
from art.attacks.evasion import FastGradientMethod, DeepFool, ProjectedGradientDescent, SquareAttack, ElasticNet, SignOPTAttack
from art.estimators.classification import PyTorchClassifier


"""
    attack pipeline.py use the function of attack.py in the adversarial attack folder
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getConfigs(datasetName, classifierName, attackName=None):
    config = utils.loadconfigYaml("/equilibrium/seide/generative_models/silva/Project-Diffusion-Defense/config.yaml")
    
    configAttack = config['attacks'][attackName]
    configAttackGeneral = config['attacks']['general']
    configData = config['dataset'][datasetName]
    configModel = config['classifier'][classifierName]

    return configAttack, configAttackGeneral, configData, configModel

def createClassifier(submodel, configsmodel):

    r""" Create a pytorch classifier to use ART attacks

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(submodel.parameters(), lr=configsmodel['learning_rate'], momentum=configsmodel['momentum'])

    classifier = PyTorchClassifier(model=submodel,clip_values=(0, 1),loss=criterion,optimizer=optimizer,input_shape=(1, 28,28),nb_classes=10,)

    return classifier

def createAdversarialData(attack, testset, batch_size):
         
    for i, (images, label) in enumerate(testset):
        if i == 0:
            x_test_adv = torch.tensor(attack.generate(x = images.cpu().detach().numpy()))
            l = label
        else:
            x_test_adv = torch.cat ([x_test_adv,torch.tensor(attack.generate(x = images.cpu().detach().numpy()))])
            l = torch.cat ([l,label ])


    #advtestset = torch.tensor(x_test_adv)

    #dataset_ = TensorDataset(advtestset , l)
    #advtestloader = DataLoader(dataset_, batch_size = batch_size)

    return x_test_adv , l

def FGSM_Attack_CH(submodel, datasetname, classifiername, testset, batchSize, type):
    """
    FGSM ATTACK
    """
    
    print("Attack using FGSM CH....")

    for i, (images, label) in enumerate(testset):
        
        images, label = images.to(device), label.to(device)

        if i == 0:
            advtestset = fast_gradient_method(model_fn=submodel, x=images, norm=np.inf,eps=0.3).detach().clone()
            l = label
        else:
            advtestset = torch.cat ([advtestset,fast_gradient_method(model_fn=submodel, x=images, norm=np.inf,eps=0.3).detach().clone()])
            l = torch.cat ([l,label ])
    
    # create a test using the adversarial images
    #dataset_ = TensorDataset(advtestset , l)
    #advtestloader = DataLoader(dataset_, batch_size = batchSize)
    
    
    print("Attack conclude....")
    
    return advtestset , l
    
def PGD_Attack_CH(submodel, datasetname, classifiername, testset, batchSize, type):
    
    
    print("Attack using PGD CH....")
    #get configs


    for i, (images, label) in enumerate(testset):
        
        images, label = images.to(device), label.to(device)

        if i == 0:
            advtestset = projected_gradient_descent(model_fn=submodel,x=images,eps=0.3,norm=np.inf, eps_iter=0.05, nb_iter = 5).detach().clone()
            l = label
        else:
            advtestset = torch.cat ([advtestset,projected_gradient_descent(model_fn=submodel,x=images,eps=0.3,norm=np.inf, eps_iter=0.05, nb_iter = 5).detach().clone()])
            l = torch.cat ([l,label])
    
    # create a test using the adversarial images
    #dataset_ = TensorDataset(advtestset , l)
    #advtestloader = DataLoader(dataset_, batch_size = batchSize)

    print("Attack conclude....")
    
    return advtestset , l

def FGSM_Attack(submodel, datasetname, classifiername, testset, batchSize, type):
    
    print("Attack using FGSM....")
    #get configs
    configAttack, _, _, configModel = getConfigs(datasetname, type, "fgsm")

    # Defining the model to attack
    classifier = createClassifier(submodel, configModel)
    
    #Defining the attack
    attack = FastGradientMethod(estimator=classifier, eps=configAttack['eps'], norm=configAttack['norm'])

    # Attaching testset and create a dataloader with adv_images
    advtestloader, l = createAdversarialData(attack, testset, batchSize)
    
    print("Attack conclude....")
    
    return advtestloader, l

def SA_Attack(submodel, datasetname, classifiername, testset, batchSize,type):
    
    
    print("Attack using SA....")
    #get configs
    _, _, _, configModel = getConfigs(datasetname, type, "fgsm")

    #attack
    classifier = createClassifier(submodel, configModel)

    attack = SquareAttack(estimator=classifier)

    advtestloader, l = createAdversarialData(attack, testset, batchSize)

    print("Attack conclude....")
    
    return advtestloader, l

def SOPT_Attack(submodel, datasetname, classifiername, testset, batchSize, type):
    
    
    print("Attack using SOPT....")
    #get configs
    _, _, _, configModel = getConfigs(datasetname, type, "fgsm")

    #attack

    classifier = createClassifier(submodel, configModel)

    attack = SignOPTAttack(estimator=classifier, targeted=False, max_iter=100, batch_size=100)

    advtestloader, l = createAdversarialData(attack, testset, batchSize)

    print("Attack conclude....")

    return advtestloader, l

def EOTPGD_Attack(submodel, datasetname, classifiername, testset, batchSize, type):
    
    print("Attack using EOTPGD....")

    attack = EOTPGD(model=submodel, eps = 0.3, steps=20, eot_iter=10)

    submodel = submodel.to(device)

    for i, (images, label) in enumerate(testset):
        images, label = images.to(device), label.to(device)
        output = submodel(images)
        _, predicated = torch.max(output.data, 1)

        if i == 0:
            x_test_adv = attack(images, predicated)
            l = label
        else:
            x_test_adv = torch.cat ([x_test_adv,attack(images, predicated)])
            l = torch.cat ([l,label ])

    print("Attack conclude....")
    
    return x_test_adv , l

def AA_Attack(submodel, datasetname, classifiername, testset, batchSize, type):
    
    print("Attack using AutoAttack....")
    
    attack = AutoAttack(model=submodel, eps = 0.3)
    submodel = submodel.to(device)

    for i, (images, label) in enumerate(testset):
        images, label = images.to(device), label.to(device)
        output = submodel(images)
        _, predicated = torch.max(output.data, 1)

        if i == 0:
            x_test_adv = attack(images, predicated)
            l = label
        else:
            x_test_adv = torch.cat ([x_test_adv,attack(images, predicated)])
            l = torch.cat ([l,label ])

    print("Attack conclude....")
    
    return x_test_adv.clone().detach() , l

def DF_Attack(submodel, datasetname, classifiername,testset, batchSize, type):
    
    print("Attack using DF....")

    #get configs
    _, _, _, configModel = getConfigs(datasetname, type, 'cw')

    # Defining the model to attack
    classifier = createClassifier(submodel, configModel)
    
    #Defining the attack
    attack = DeepFool(classifier=classifier, batch_size=32)

    # Attaching testset and create a dataloader with adv_images
    advtestloader, l = createAdversarialData(attack, testset, batchSize)


    print("Attack conclude....")
    
    return advtestloader, l

def PGD_Attack(submodel, datasetname, classifiername,testset, batchSize, type):    

    print("Attack using PGD....")
    #get configs
    _, _, _, configModel = getConfigs(datasetname, type, 'cw')

    # Defining the model to attack
    classifier = createClassifier(submodel, configModel)
    
    #Defining the attack
    attack = ProjectedGradientDescent(classifier,norm=np.inf,eps=0.3,eps_step=0.01,max_iter=20,targeted=False,num_random_init=5)

    # Attaching testset and create a dataloader with adv_images
    advtestloader,l = createAdversarialData(attack, testset, batchSize)

    print("Attack conclude....")

    return advtestloader, l

def EN_Attack(submodel, datasetname, classifiername,testset, batchSize, type):    

    print("Attack using Elastic NET....")
    #get configs
    _, _, _, configModel = getConfigs(datasetname, type, 'cw')

    
    # Defining the model to attack
    classifier = createClassifier(submodel, configModel)
    
    #Defining the attack
    attack = ElasticNet(classifier,batch_size=batchSize)

    # Attaching testset and create a dataloader with adv_images
    advtestloader,l = createAdversarialData(attack, testset, batchSize)

    print("Attack conclude....")

    return advtestloader, l


