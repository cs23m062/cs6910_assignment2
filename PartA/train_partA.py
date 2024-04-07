import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import torchmetrics
import lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
import argparse
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

#inherits the module class present in pytorch
class ConvolutionNeuralNet(nn.Module):
    #function to choose appropriate activation function
    def activation_function(self,activ):
        if activ == 'ReLU' :
            return nn.ReLU()
        elif activ == 'GELU':
            return nn.GELU()
        elif activ == 'SiLU':
            return nn.SiLU()
        elif activ == 'Mish':
            return nn.Mish()
        elif activ == 'CELU':
            return nn.CELU()
        else:
            return nn.Tanh()

    #function which updates the depth of each layer as per user's configuration
    def update_depth(self,k,org):
        if(org=='double'):
            return 2*k
        elif(org=='half' and k>1):
            return k//2
        else:
            return k

    #Constructor to intialize the model
    def __init__(self,config):
        super(ConvolutionNeuralNet, self).__init__()

        nfilters = config['filter_per_layer']
        print(nfilters)
        self.padding = config['padding']        #initialize the padding of your model, same is applied in every layer
        self.stride = config['stride']          #initialize the stride of your model, same is applied in every pooling layer

        #Layer 1 => Convolution_1 + Activation_1 + pooling_1
        self.conv1 = nn.Conv2d(3,nfilters,kernel_size = config['filter_length'], stride=1, padding=self.padding)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (112 - config['filter_length'] + 2*self.padding) + 1
        if(config['batch_normalization'] == 'yes'):
            self.bn1 = nn.BatchNorm2d(nfilters)
        self.act1 = self.activation_function(config['activation'])
        self.pool1 = nn.MaxPool2d(kernel_size=config['filter_length'], stride=self.stride)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'])//self.stride + 1
        print(height,nfilters)

        #update the number of filters as per user's choice
        last = nfilters
        nfilters = self.update_depth(nfilters,config['filter_organisation'])

        #Layer 2 => Convolution_2 + Activation_2 + pooling_2
        self.conv2 = nn.Conv2d(last,nfilters,kernel_size=config['filter_length'], stride=1, padding=self.padding)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'] + 2*self.padding) + 1
        if(config['batch_normalization'] == 'yes'):
            self.bn2 = nn.BatchNorm2d(nfilters)
        self.act2 = self.activation_function(config['activation'])
        self.pool2 = nn.MaxPool2d(kernel_size=config['filter_length'], stride=self.stride)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'])//self.stride + 1
        print(height,last,nfilters)

        #update the number of filters as per user's choice
        last = nfilters
        nfilters = self.update_depth(nfilters,config['filter_organisation'])

        #Layer 3 => Convolution_3 + Activation_3 + pooling_3
        self.conv3 = nn.Conv2d(last,nfilters,kernel_size=config['filter_length'], stride=1, padding=self.padding)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'] + 2*self.padding) + 1
        if(config['batch_normalization'] == 'yes'):
            self.bn3 = nn.BatchNorm2d(nfilters)
        self.act3 = self.activation_function(config['activation'])
        self.pool3 = nn.MaxPool2d(kernel_size=config['filter_length'], stride=self.stride)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'])//self.stride + 1
        print(height,last,nfilters)

        #update the number of filters as per user's choice
        last = nfilters
        nfilters = self.update_depth(nfilters,config['filter_organisation'])

        #Layer 4 => Convolution_4 + Activation_4 + pooling_4
        self.conv4 = nn.Conv2d(last,nfilters,kernel_size=config['filter_length'], stride=1, padding=self.padding)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'] + 2*self.padding) + 1
        if(config['batch_normalization'] == 'yes'):
            self.bn4 = nn.BatchNorm2d(nfilters)
        self.act4 = self.activation_function(config['activation'])
        self.pool4 = nn.MaxPool2d(kernel_size=config['filter_length'], stride=self.stride)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'])//self.stride + 1
        print(height,last,nfilters)

        #update the number of filters as per user's choice
        last = nfilters
        nfilters = self.update_depth(nfilters,config['filter_organisation'])

        #Layer 5 => Convolution_5 + Activation_5 + pooling_5
        self.conv5 = nn.Conv2d(last,nfilters,kernel_size=config['filter_length'], stride=1, padding=self.padding)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'] + 2*self.padding) + 1
        if(config['batch_normalization'] == 'yes'):
            self.bn5 = nn.BatchNorm2d(nfilters)
        self.act5 = self.activation_function(config['activation'])
        self.pool5 = nn.MaxPool2d(kernel_size=config['filter_length'], stride=self.stride)
        # ===========> Calculate the size of the resultant image  <=======================
        height = (height - config['filter_length'])//self.stride + 1

        print(height,last,nfilters)
        #Fully connected layer
        self.fc1 = nn.Linear(nfilters * height * height, config['dense_size'])
        self.act6 = self.activation_function(config['activation'])
        self.drop = nn.Dropout(p=config['drop'])
        self.fc2 = nn.Linear(config['dense_size'], 10)  # Our 10 classes for classification
        #self.soft = nn.Softmax(dim=1)
        
        self.config = config
    #forward pass of model :=> general flow : convolution -> activation -> maxpool -> dense layer(dropout if give) => softmax
    def forward(self, x):
        #convolution layer starts
        # Conv layer 1
        x = self.conv1(x)
        #apply batch normalization if specified
        if(self.config['batch_normalization']=='yes'):
            x = self.bn1(x)
        x = self.pool1(self.act1(x))

        # Conv layer 2
        x = self.conv2(x)
        #apply batch normalization if specified
        if(self.config['batch_normalization']=='yes'):
            x = self.bn2(x)
        x = self.pool2(self.act2(x))

        # Conv layer 3
        x = self.conv3(x)
        #apply batch normalization if specified
        if(self.config['batch_normalization']=='yes'):
            x = self.bn3(x)
        x = self.pool3(self.act3(x))

        # Conv layer 4
        x = self.conv4(x)
        #apply batch normalization if specified
        if(self.config['batch_normalization']=='yes'):
            x = self.bn4(x)
        x = self.pool4(self.act4(x))

        # Conv layer 5
        x = self.conv5(x)
        #apply batch normalization if specified
        if(self.config['batch_normalization']=='yes'):
            x = self.bn5(x)

        # ===========> Dense Layer starts <=============
        x = self.pool5(self.act5(x))
        x = x.reshape(x.size(0), -1)
        x = self.act6(self.fc1(x))
        x = self.drop(x)
        #x = self.soft(self.fc2(x))
        # ============> softmax layer <==============
        x = self.fc2(x)
        return x

#use the optimizer function as necessary
def optimizer_function(name,lr,model):
    if(name=='adam'):
        return optim.Adam(model.parameters(), lr=lr)
    elif(name=='nadam'):
        return optim.NAdam(model.parameters(),lr=lr)
    elif(name=='sgd'):
        return optim.SGD(model.parameters(), lr=lr,momentum=0.8,nesterov=True)
    else:
        return optim.RMSprop(model.parameters(), lr=lr,alpha=0.9) 

#Defining the pytorch lightning model
class LightningCNN(L.LightningModule):
    def __init__(self,model,learning_rate):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)        #calculates training accuracy
        self.valid_acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)        #calculates validation accuracy
        self.test_acc =  torchmetrics.Accuracy(task="multiclass",num_classes=10)        #calculates test accuracy
    
    #forward pass
    def forward(self, x): 
        return self.model(x)
    
    #Initializes the optimizer
    def configure_optimizers(self):
        optimizer = optimizer_function(self.model.config['optimizer'],self.model.config['learning_rate'],self.model)
        return optimizer
    
    #This function gets called when training starts
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)    #predicts outputs
        loss = self.criterion(outputs, labels)    #calcualte loss

        preds = torch.argmax(outputs, dim=1)    #calculate correct predictions
        self.train_acc(preds, labels)
        #logs the training loss at the end of epoch
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        #logs the training accuracy at the end of epoch
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    #This function gets called when validation starts
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)         #predicts outputs
        loss = self.criterion(outputs, labels)    #calcualte loss

        preds = torch.argmax(outputs, dim=1)    #calculate correct predictions
        self.valid_acc(preds, labels)
        #logs the validation loss at the end of epoch
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
         #logs the validation accuracy at the end of epoch
        self.log('val_acc', self.valid_acc.compute(), prog_bar=True, logger=True)
        self.valid_acc.reset()

    #same as above these are for test steps
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        preds = torch.argmax(outputs, dim=1)
        self.test_acc(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        # Reset metrics for next epoch
        self.test_acc.reset()
        

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Load the iNaturalist dataset
dataset = datasets.ImageFolder(root='/kaggle/input/lightnature/nature_12K/inaturalist_12K/train', transform=transform)

# Split the dataset into train and test
train_len = int(0.8 * len(dataset))  # Assuming 80% for training (integer division)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, lengths=[train_len, test_len])

vali_dataset = datasets.ImageFolder(root='/kaggle/input/lightnature/nature_12K/inaturalist_12K/val', transform=transform)


wandb.login() #key='fc18454f0555cdcc5d98e84dfb27e127061d3d8b'

config = {
    'epochs': 5,
    'batch_size':32,
    'activation':'Mish',
    'filter_per_layer' :64,
    'filter_length' : 3,
    'padding' : 0,
    'batch_normalization' : 'yes',
    'filter_organisation' : 'same',
    'dense_size' : 1024,
    'drop' : 0,
    'stride' : 1,
    'optimizer' :'sgd',
    'learning_rate' : 0.001
}

def main(args):

    config['epochs'] = args.epochs
    config['activation'] = args.activation
    config['batch_size'] = args.batch_size
    config['filter_per_layer'] = args.filter_per_layers
    config['filter_length'] = args.filter_length
    config['batch_normalization'] = args.batchnorm
    config['filter_organisation'] = args.filterorg
    config['dense_size'] = args.dense_size
    config['drop'] = args.dropout
    config['stride'] = args.stride
    config['optimizer'] = args.optimizer
    config['learning_rate'] = args.learning_rate

    wandb.init(project=args.wandb_project,config=config)
    wandb_logger = WandbLogger(project=args.wandb_project,log_model='all')
    #config = wandb.config
    #load datasets
    global train_loader,vali_loader
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    vali_loader = DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True)
    
        
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')
    model = ConvolutionNeuralNet(config)
    image_classifier = LightningCNN(model,config['learning_rate'])
    trainer = L.Trainer(max_epochs=config['epochs']) 
    # Train the model
    trainer.fit(image_classifier, train_loader, vali_loader)
    #predict the outputs
    trainer.test(image_classifier,dataloaders=DataLoader(test_dataset,batch_size=config['batch_size']))
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep_LearingAssignment2_CS23M062 -command line arguments")
    parser.add_argument("-wp","--wandb_project", type=str, default ='Shubhodeep_CS6190_LightningDeepLearing_Assignment2', help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we","--wandb_entity", type=str, default ='shubhodeepiitm062',help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-e","--epochs",type=int,default = 5,help ='Number of epochs to train convolutional neural network.')
    parser.add_argument("-b","--batch_size",type=int,default = 32,help='Batch size used to train neural network.')
    parser.add_argument('-o','--optimizer',type=str,default='sgd',help='choices: ["sgd","rmsprop", "adam", "nadam"]')
    parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning rate used to optimize model parameters')
    parser.add_argument('-fpl','--filter_per_layers',type=int,default=64,help='Number of filters to be used per convolution layer')
    parser.add_argument('-ds','--dense_size',type=int,default=1024,help='Number of hidden neurons in a fully connected layer.')
    parser.add_argument('-a','--activation',type=str,default='Mish',help = 'choices: ["Mish", "ReLU", "GELU", "CELU","SiLU","Tanh"]')
    parser.add_argument('-d','--dropout',type=str,default=0,help="Dropout probability in dense layer")
    parser.add_argument('-s',"--stride",type=int,default=1,help="length of stride in maxpooling layer")
    parser.add_argument('-bn','--batchnorm',type=str,default='yes',help="yes if want to use batch normalization else no")
    parser.add_argument('-fo',"--filterorg",type=str,default='same',help="same will keep same number of filters every layer, double doubles and half halves every layer(maxpool + conv)")
    parser.add_argument('-fl',"--filter_length",type=int,default=3,help="length of the filter")
    args = parser.parse_args()
    main(args)
