import torch
from torch import optim
import wandb
from classifiers.classifier import *
from tqdm import tqdm
from statistics import mode
from dataset import * 
from torch.utils.data import DataLoader , TensorDataset

import argparse

# for adv training
from attackpipeline import *

# Hyperparamters for the training pipeline
hyperparameters = {
        'LR':             1e-2,
        'EPOCHS':         20,
        'BATCH_SIZE':     128,
        'MOMENTUM':       0.9,
        'CLASSES':        10,
        'DATASET':        'MNIST',
        'CLASSIFIER':     'classifier_b',
        'MODEL_TITLE':    'name',
        'LOG':            'disabled'
}


#check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    parser = argparse.ArgumentParser(description='DiffDefence: Train classifier module')
    
    parser.add_argument("--lr", type=float, default=1e-2, help='learning rate')
    parser.add_argument("--dataset", type=str, default="MNIST", help='dataset (MNIST-KMNIST)')
    parser.add_argument("--epochs", type=int, default="20", help='Training epochs')
    parser.add_argument("--batch_size", type=int, default="128", help='Batch size')
    parser.add_argument("--momentum", type=float, default="0.9", help='momentum')
    parser.add_argument("--classes", type=int, default="10", help='dataset class')
    parser.add_argument("--model_title", type=str, default="modelname", help='Model name') 
    parser.add_argument("--log", type=str, default="disabled", help='Wandb logging')
    parser.add_argument("--classifier", type=str, default="classifier_a", help='classifier_a or classifier_b') 
    parser.add_argument("--adv_train", type=bool , default=False, help='Adversarial training')

    args = parser.parse_args()

    hyperparameters["LR"] = args.lr
    hyperparameters["DATASET"] = args.dataset
    hyperparameters["EPOCHS"] = args.epochs
    hyperparameters["BATCH_SIZE"] = args.batch_size
    hyperparameters["MOMENTUM"] = args.momentum
    hyperparameters["CLASSES"] = args.classes
    hyperparameters["MODEL_TITLE"] = args.model_title
    hyperparameters["LOG"] = args.log
    hyperparameters['CLASSIFIER'] = args.classifier

    
    trainloader , testloader = getData(datasetname=hyperparameters["DATASET"], typedata="both", batch_size=hyperparameters["BATCH_SIZE"])

    model_pipeline(classifierName=hyperparameters["CLASSIFIER"],
                   datasetname=hyperparameters["DATASET"],
                   trainloader=trainloader,
                   testloader=testloader,
                   adv_train = args.adv_train)



def model_pipeline(classifierName, datasetname, trainloader, testloader, adv_train):

    
    with wandb.init(project="classifier-diffusion-defense", config=hyperparameters, mode = hyperparameters['LOG']):
        #access all HPs through wandb.config
        config = wandb.config

        #make the model, data and optimization problem
        model, criterion, optimizer= create(config, classifierName)

        #train the model
        if adv_train == True:
            print(f"Adversarial training on {config['CLASSIFIER']}")
            adversarial_train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader)
        else:
            print(f"Training {config['CLASSIFIER']}")
            train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader)

        #test the model
        print(f"Accuracy test: {test(model, testloader)}%")
        
    return model

def create(config, classifierName):

    #Create a model
    model = classifiers[classifierName].to(device)
    
    #Create the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM)

    return model, criterion, optimizer

def adversarial_train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader):
    r"""
    Method that implement adversarial training
    """
    if wandb.run is not None:
        wandb.watch(model, criterion, log="all", log_freq=10)
    
    example_ct,batch_ct  = 0, 0

    for epoch in range(config.EPOCHS):   # loop over the dataset multiple times
        pbar = tqdm(trainloader, leave=False)   
        for _, (images, labels) in enumerate(pbar):
            
            #create adversarial samples
            dl_to_adv = DataLoader(TensorDataset(images, labels), batch_size=images.shape[0])
            input, l = FGSM_Attack_CH(submodel = model, datasetname=datasetname, classifiername=classifierName, testset=dl_to_adv, batchSize = 32)
            input, images, l, labels = input.to(device) , images.to(device), l.to(device), labels.to(device)
            
            images = torch.cat((images, input), 0)
            labels = torch.cat((labels, l), 0)

            loss = train_batch(images, labels, model, optimizer, criterion)

            example_ct += len(images)
            batch_ct += 1
                            
            pbar.set_postfix(MSE=loss.item())
        
        torch.save(model.state_dict(), f"./pretrained/{config['DATASET']}/{config['MODEL_TITLE']}.pt")
        train_log(loss, example_ct, epoch)
    return model

def train(model, trainloader, criterion, optimizer, config, classifierName, datasetname, testloader):

    #telling wand to watch
    if wandb.run is not None:
        wandb.watch(model, criterion, log="all", log_freq=10)
    
    example_ct,batch_ct  = 0, 0

    for epoch in range(config.EPOCHS):  # loop over the dataset multiple times
        pbar = tqdm(trainloader, leave=False)   
        for _, (images, labels) in enumerate(pbar):
                        
            loss = train_batch(images, labels, model, optimizer, criterion)

            example_ct += len(images)
            batch_ct += 1
                            
            pbar.set_postfix(MSE=loss.item())
        
        torch.save(model.state_dict(), f"./pretrained/{config['DATASET']}/{config['MODEL_TITLE']}.pt")

        train_log(loss, example_ct, epoch)
    return model
 

def train_batch(images, labels,model, optimizer, criterion):

    #insert data into cuda if available
    images,labels = images.to(device), labels.to(device)
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    #backward pass
    loss.backward()

    #step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)

    if wandb.run is not None:
        wandb.log({"epoch":epoch, "loss":loss}, step=example_ct)
    
def test(model, test_loader):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            oututs = model(images)
            _, predicated = torch.max(oututs.data, 1)
            total += labels.size(0)

            correct += (predicated == labels).sum().item()

        return correct/total


def testadv(model_, images_, labels_, n_):

    model_.eval()

    with torch.no_grad():
        correct, total = 0, n_
        predicts, l = [], []
        for i in range(n_):
            images, labels = images_[i].to(device), labels_[i].to(device)
            oututs = model_(images[None,:,:,:])
            
            _, predicated = torch.max(oututs.data, 1)
            predicts.append(predicated.item())
            l.append(labels.item())

            correct += (predicated == labels).sum().item()

        if wandb.run is not None:
            wandb.log({"test_accuracy":correct/total})
        
        return 100*correct/total




if __name__ == '__main__':

    main()





   