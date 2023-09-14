import torch
from torch import nn
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights, vgg16
from torchvision import datasets, transforms, utils
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
import lightning.pytorch as L

def mnist_dataloader(type:str, batch_size:int):

    if type == 'train':
        train_dataset = datasets.MNIST(
            root='./data',      
            train=True,          
            download=True,    
            transform=transforms.ToTensor()
        )

        train_dataset, val_dataset = random_split(
            dataset=train_dataset,
            lengths=[int(len(train_dataset)*0.8), int(len(train_dataset)*0.2)],
            generator=torch.Generator().manual_seed(777)
        )
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)
        val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle=True)

        return train_loader, val_loader
    
    elif type == 'test':
        test_dataset = datasets.MNIST(
            root='./data',     
            train=False,         
            download=True,       
            transform=transforms.ToTensor()
        )
        test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle=False)

        return test_loader
    
    else:
        print('No this type!')

class MNISTModel(L.LightningModule):
    def __init__(self, args, num_classes):
        super().__init__()

        self.save_hyperparameters()
        self.args = args
        self.num_classes = num_classes
        self.model = build_model(args, num_classes)
        if args.loss_function in('CrossEntropyLoss', 'BCE', 'BCEWithLogitsLoss') :
            self.loss_function = getattr(torch.nn, args.loss_function)()
        else:
            raise NameError('The loss function is not supported.')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_function(output, y)
        _, pred = torch.max(output, 1)
        acc = accuracy_score(pred.cpu(), y.cpu())
        #  on_step: Logs the metric at the current step.
        #  on_epoch =True:Automatically accumulates and logs at the end of the epoch.
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_function(output, y)
        _, pred = torch.max(output, 1)
        acc = accuracy_score(pred.cpu(), y.cpu())

        self.log("validation_loss", loss, on_step=True, on_epoch=True) 
        self.log("validation_acc", acc, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        # loss = self.loss_function(output, y)
        _, pred = torch.max(output, 1)
        acc = accuracy_score(pred.cpu(), y.cpu())

        self.log("test_acc", acc)

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     x, y = batch
    #     y = self.model(x)
    #     return y

    def configure_optimizers(self):
        optimizer, lr_scheduler = get_hyper(self.model, self.args)
        optim_dict = {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        return optim_dict

def build_model(args, num_classes):
    if args.model_name.upper() == 'RESNET':
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        conv1_out_channels = net.conv1.out_channels
        net.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        fc_features = net.fc.in_features 
        net.fc = nn.Linear(fc_features, num_classes)
    elif args.model_name.upper() == 'VGG':
        net = vgg16()
        net.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, num_classes), nn.Softmax(dim=1))
    else:
        raise NameError('The model is not accessed.')
    
    return net

def get_hyper(model, args):

    if args.OPTIM in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
        optimizer = getattr(optim, args.OPTIM)(model.parameters(), lr=args.LEARNING_RATE, betas=(args.MOMENTUM, 0.999), weight_decay=args.WEIGHT_DECAY)
    elif args.OPTIM == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.LEARNING_RATE, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY, nesterov=True)
    elif args.OPTIM == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=args.LEARNING_RATE, momentum=args.MOMENTUM)
    elif args.OPTIM == "Adadelta":
        optimizer = optim.AdamW(model.parameters(), lr=args.LEARNING_RATE, weight_decay=args.WEIGHT_DECAY)
    else:
        raise NameError('The optimizer is not supported.')

    if args.lr_scheduler == 'stepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    elif args.lr_scheduler == 'MultiStepLR':
        optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
    elif args.lr_scheduler == 'LambdaLR':
        last_lr = 0.01  # rate (lr*last_lr)
        lr_lambda = lambda x: (1 - x / args.NUM_EPOCHS) * (1.0 - last_lr) + last_lr
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise NameError('The learning rate scheduler is not supported.')

    return optimizer, lr_scheduler


# class lightning_dataloader(pl.LightningDataModule):
#     def __init__(self, batch_size:int=64):
#         super().__init__()

#         self.batch_size = batch_size

#     # download
#     # def prepare_data(self):  
#     #     MNIST(self.data_dir, train=True, download=True)
#     #     MNIST(self.data_dir, train=False, download=True)

#     # Assign train/val datasets for use in dataloaders
#     def setup(self, stage: str):
#         self.train_dataloader, self.val_dataloader = mnist_dataloader(batch_size=self.batch_size, type='train')
#         self.test_dataloader = mnist_dataloader(batch_size=1, type='test')

#     def train_dataloader(self):
#         return self.train_dataloader

#     def val_dataloader(self):
#         return self.val_dataloader

#     def test_dataloader(self):
#         return self.test_dataloader

    
def main():
    
    # data_loader = lightning_dataloader(batch_size=64)
    # train_dataloader = data_loader.train_dataloader()
    # print(type(train_dataloader))

    train_loader, val_loader, test_loader = mnist_dataloader(batch_size=64)
    print(len(train_loader))

if __name__ == '__main__':
    main()
    