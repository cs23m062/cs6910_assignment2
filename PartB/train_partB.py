import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import lightning as L
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
import re
import wandb
import argparse


# Define the LightningModule
class BestWorldModels(L.LightningModule):
    def __init__(self,learning_rate,freeze=3,num_classes=10,aux_logits=True):
        super().__init__()
        
        self.lr = learning_rate
        self.save_hyperparameters()
        self.backbone = None

        self.backbone = models.googlenet(weights='DEFAULT')
        
        self.freeze = freeze  # Store the freeze value
        for name, param in self.backbone.named_parameters():
            match = re.search(r'\d+', name.split('.')[0])
            if match and int(match.group()) < freeze:
                param.requires_grad = False
        
        # init a pretrained resnet
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        in_features = self.backbone.fc.in_features
        self.classifier = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass",num_classes=10)
        self.test_acc =  torchmetrics.Accuracy(task="multiclass",num_classes=10)

    def forward(self, x):
        #with torch.no_grad():
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.valid_acc(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self,batch,batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.test_acc(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute(), prog_bar=True)
        self.test_acc.reset()


    def on_validation_epoch_end(self):
        self.log('val_acc', self.valid_acc.compute(), prog_bar=True, logger=True)
        self.valid_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4747786223888397,0.4644955098628998,0.3964916169643402],std=[0.2389, 0.2289,0.2422]),
])

# Load the iNaturalist dataset
dataset = datasets.ImageFolder(root='/kaggle/input/lightnature/nature_12K/inaturalist_12K/train', transform=transform)

# Split the dataset into train and test
train_len = int(0.8 * len(dataset))  # Assuming 80% for training (integer division)
test_len = len(dataset) - train_len
train_dataset, test_dataset = random_split(dataset, lengths=[train_len, test_len])
vali_dataset = datasets.ImageFolder(root='/kaggle/input/lightnature/nature_12K/inaturalist_12K/val', transform=transform)

config = {
    'epochs': 10,
    'batch_size':16,
    'learning_rate' : 0.001,
    'freeze' : 10,
}

def main(args):
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['freeze'] = args.freeze
    config['learning_rate'] = args.learning_rate
    global train_loader,vali_loader

    wandb.init(project=args.wandb_project,config=config)
    wandb_logger = WandbLogger(project=args.wandb_project,log_model='all')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    vali_loader = torch.utils.data.DataLoader(vali_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['batch_size'], shuffle=False)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')

    train_loader = DataLoader(train_dataset,config['batch_size'], shuffle=True)
    vali_loader = DataLoader(vali_dataset,config['batch_size'], shuffle=True)
    model = BestWorldModels(learning_rate=config['learning_rate'],freeze=config['freeze'])
    # Instantiate the Trainer
    trainer = L.Trainer(max_epochs=config['epochs'],logger=wandb_logger)
    # Train the model
    trainer.fit(model, train_loader,vali_loader)
    trainer.test(model, dataloaders=test_loader)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep_LearingAssignment2_CS23M062 -command line arguments")
    parser.add_argument("-wp","--wandb_project", type=str, default ='Shubhodeep_CS6190_LightningDeepLearing_Assignment2', help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we","--wandb_entity", type=str, default ='shubhodeepiitm062',help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-e","--epochs",type=int,default = 5,help ='Number of epochs to train convolutional neural network.')
    parser.add_argument("-b","--batch_size",type=int,default = 32,help='Batch size used to train neural network.')
    parser.add_argument('-lr','--learning_rate',type=float,default=0.001,help='Learning rate used to optimize model parameters')
    parser.add_argument('-fz','--freeze',type=int,default=5,help='choices: [0-15]')

    args = parser.parse_args()
    main(args)