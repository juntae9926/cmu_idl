import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as ttf

import os
import os.path as osp

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score
import numpy as np

import timm

batch_size = 128
lr = 0.01
epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transfer_NN(nn.Module):
    def __init__(self,encoder,classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_NN, self).__init__()

        self.base = encoder

        # Transfer to another task layer
        self.cls_layer = nn.Linear(768, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False
        self.init_weights()
    
    def load_from_pretrain(self, pretrained_path):
        print('-'*80)
        print("Loading pretrained model %s" % pretrained_path)
        print('-'*80)        
        package = torch.load(pretrained_path)
        self.base.load_state_dict(package['model'])

        
    def init_weights(self):
        self.init_layer(self.cls_layer)
        
    def forward(self, x, return_feats=False):
        """
        What is return_feats? It essentially returns the second-to-last-layer
        features of a given image. It's a "feature encoding" of the input image,
        and you can use it for the verification task. You would use the outputs
        of the final classification layer for the classification task.

        You might also find that the classification outputs are sometimes better
        for verification too - try both.
        """
        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out

    def forward(self, x, return_feats=False):
        """Input: (batch_size, data_length)
        """
        feats = self.base(x)
        out = self.cls_layer(feats)
       
        if return_feats:
            return feats
        else:
            return out

    def init_layer(self,layer):
        """Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
    
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
                

class Network(nn.Module):

    def __init__(self, num_classes=7000):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(6),
            ) 
        
        self.cls_layer = nn.Linear(512, num_classes)
    
    def forward(self, x, return_feats=False):
        feats = self.backbone(x)
        B,Ch,_,_ = feats.shape
        feats = feats.reshape(B,Ch)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out

class ClassificationTestSet(Dataset):
    # It's possible to load test set data using ImageFolder without making a custom class.
    # See if you can think it through!

    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))


if __name__ == "__main__":
    DATA_DIR = "/home/mmlab/idl/HW2P2/hw2p2-data"
    TRAIN_DIR = osp.join(DATA_DIR, "classification/classification/train")
    VAL_DIR = osp.join(DATA_DIR, "classification/classification/dev")
    TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")

    train_transforms = [ttf.RandomResizedCrop((224, 224)), ttf.RandomHorizontalFlip(), ttf.ToTensor()]
    val_transforms = [ttf.ToTensor()]
    train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR,
                                                    transform=ttf.Compose(train_transforms))
    val_dataset = torchvision.datasets.ImageFolder(VAL_DIR,
                                                transform=ttf.Compose(val_transforms))


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=2)

    #model = Network()

    tempmodel = timm.create_model("convnext_tiny", pretrained=False)
    del tempmodel.head.fc
    #print(model)

    model = Transfer_NN(tempmodel, classes_num=7000, freeze_base=False)
    model = torch.nn.DataParallel(model)

    model.cuda()

    num_trainable_parameters = 0
    for p in model.parameters():
        num_trainable_parameters += p.numel()
    print("Number of Params: {}".format(num_trainable_parameters))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
    #scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200, gamma=0.99)

    for epoch in range(epochs):
        # Quality of life tip: leave=False and position=0 are needed to make tqdm usable in jupyter
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

        num_correct = 0
        total_loss = 0
        # Train

        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x = x.cuda()
            y = y.cuda()
    
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            pred = torch.argmax(outputs, dim=1)
            num_correct += pred.eq(y).sum().item()
            total_loss += float(loss)

            # tqdm lets you add some details so you can monitor training as you train.
            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

            scheduler.step() # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

            batch_bar.update() # Update tqdm bar
        batch_bar.close() # You need this to close the tqdm bar

        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            epochs,
            100 * num_correct / (len(train_loader) * batch_size),
            float(total_loss / len(train_loader)),
            float(optimizer.param_groups[0]['lr'])))

        # Validation 
        model.eval()
        batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')
        num_correct = 0
        for i, (x, y) in enumerate(val_loader):

            x = x.cuda()
            y = y.cuda()

            with torch.no_grad():
                outputs = model(x)

            num_correct += int((torch.argmax(outputs, dim=1) == y).sum())
            batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)))

            batch_bar.update()
            
        batch_bar.close()
        print("Validation: {:.04f}%".format(100 * num_correct / len(val_dataset)))
        torch.save(model.module.state_dict(), '/home/mmlab/idl/HW2P2/' +  'model_1.pt')


    test_dataset = ClassificationTestSet(TEST_DIR, ttf.Compose(val_transforms))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=1)

    model.eval()

    res = []
    for i, (x) in enumerate(tqdm(test_loader)):
        x = x.cuda()
        with torch.no_grad():
            outputs = model(x)
            pred = torch.argmax(outputs, dim=1)
            res.extend(pred.tolist())

    with open("/home/mmlab/idl/HW2P2/submission.csv", "w+") as f:
        f.write("id,label\n")
        for i in range(len(test_dataset)):
            f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))

