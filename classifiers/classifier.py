import torch.nn as nn


class classifier_a(nn.Module):
    def __init__(self):
        super(classifier_a, self).__init__()
        self.relu = nn.ReLU()
        #self.sm = nn.Softmax()
        #self.conv0 = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=[5,5],stride=1) 
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64, kernel_size=[5,5],stride=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=[5,5],stride=2) #10x10x64
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(out_features=128, in_features=6400) 
        self.fc2 = nn.Linear(out_features=10, in_features=128)

    def forward(self, x):
        #x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout1(x)
        
        x = x.view(-1, 6400) # 6400 = 10x10x64

        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        #x = self.sm(x)
       
        return x


class classifier_b(nn.Module):
    def __init__(self):
        super(classifier_b, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64, kernel_size=[8,8],stride=2) #11x11x64
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[6,6],stride=2) #3x3x128
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[5,5],stride=1, padding=1)#1x1x128

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(out_features=10, in_features=128)

    def forward(self, x):

        x = self.dropout1(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.dropout2(x)

        x = x.view(-1, 128)
        
        x = self.fc1(x)

        return x

"""
    Classifier F, D, C, E NOT USED in the project!  
"""

class classifier_f(nn.Module):
    def __init__(self):
        super(classifier_f, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64, kernel_size=[8,8],stride=2) #11x11x64
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=[6,6],stride=2) #3x3x128
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=[5,5],stride=1, padding=1)#1x1x128

        self.fc1 = nn.Linear(out_features=10, in_features=128)

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(-1, 128)
        
        x = self.fc1(x)

        return x

class classifier_c(nn.Module):
    def __init__(self):
        super(classifier_c, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=128, kernel_size=[3,3],stride=1) #26x26x128
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=[3,3],stride=2) #12x12x64
        
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(out_features=128, in_features=9216) #9212 = 12*12*64
        self.fc2 = nn.Linear(out_features=10, in_features=128)

    def forward(self, x):

        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout1(x)
        
        x = x.view(-1, 9216)
        
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

class classifier_d(nn.Module):
    def __init__(self):
        super(classifier_d, self).__init__()
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(out_features=200, in_features=(28*28))
        self.fc2 = nn.Linear(out_features=200, in_features=200)
        self.fc3 = nn.Linear(out_features=10, in_features=200)

    def forward(self, x):
        #x.shape = [B,C,H,W]
        x = x.view(-1, x.size(2)*x.size(2))
        
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)        
        x = self.relu(self.fc2(x))
        x = self.dropout1(x)        
        x = self.fc3(x)

        return x

class classifier_e(nn.Module):
    def __init__(self):
        super(classifier_e, self).__init__()
        self.relu = nn.ReLU()        
        self.fc1 = nn.Linear(out_features=200, in_features=(28*28))
        self.fc2 = nn.Linear(out_features=200, in_features=200)
        self.fc3 = nn.Linear(out_features=10, in_features=200)

    def forward(self, x):
        #x.shape = [B,C,H,W]
        x = x.view(-1, x.size(2)*x.size(2))
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# dict used to get classifiers 
classifiers = {
    'classifier_a': classifier_a(),'classifier_b': classifier_b(),     
    'classifier_c': classifier_c(),'classifier_d': classifier_d(),
    'classifier_e': classifier_e(), 'classifier_f': classifier_f()
}

def getClassifier(classifierName):
    return classifiers[classifierName]