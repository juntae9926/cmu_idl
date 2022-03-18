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

from resnet import resnet50

# Hyperparameters
batch_size = 128
lr = 0.01
epochs = 150

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

DATA_DIR = "/home/mmlab/idl/HW2P2/hw2p2-data"
TRAIN_DIR = osp.join(DATA_DIR, "classification/classification/train") # This is a smaller subset of the data. Should change this to classification/classification/train
VAL_DIR = osp.join(DATA_DIR, "classification/classification/dev")
TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")
best_path = "/home/mmlab/idl/HW2P2/result/model"

train_transforms = [ttf.RandomResizedCrop((224, 224), scale=(0.1, 1), ratio=(0.5, 2)), ttf.GaussianBlur(3, sigma=(0.1, 2.0)), ttf.RandomHorizontalFlip(), ttf.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5), ttf.ToTensor()]
val_transforms = [ttf.ToTensor()]

train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR, transform=ttf.Compose(train_transforms))
val_dataset = torchvision.datasets.ImageFolder(VAL_DIR, transform=ttf.Compose(val_transforms))


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=1)

model = resnet50()
model = torch.nn.DataParallel(model)
model.cuda()

# For this homework, we're limiting you to 35 million trainable parameters, as
# outputted by this. This is to help constrain your search space and maintain
# reasonable training times & expectations
num_trainable_parameters = 0
for p in model.parameters():
    num_trainable_parameters += p.numel()
print("Number of Params: {}".format(num_trainable_parameters))

# TODO: What criterion do we use for this task?
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * epochs))
# T_max is "how many times will i call scheduler.step() until it reaches 0 lr?"

# For this homework, we strongly strongly recommend using FP16 to speed up training.
# It helps more for larger models.
# Go to https://effectivemachinelearning.com/PyTorch/8._Faster_training_with_mixed_precision
# and compare "Single precision training" section with "Mixed precision training" section
scaler = torch.cuda.amp.GradScaler()

for epoch in range(epochs):
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    num_correct = 0
    total_loss = 0
    best_acc = 0

    # Training
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.cuda()
        y = y.cuda()

        # Don't be surprised - we just wrap these two lines to make it work for FP16
        with torch.cuda.amp.autocast():     
            outputs = model(x)
            loss = criterion(outputs, y)

        # Update # correct & loss as we go
        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        total_loss += float(loss)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
        
        # Another couple things you need for FP16. 
        scaler.scale(loss).backward() # This is a replacement for loss.backward()
        scaler.step(optimizer) # This is a replacement for optimizer.step()
        scaler.update() # This is something added just for FP16

        scheduler.step() 

        batch_bar.update() 
    batch_bar.close() 

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

        num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
        batch_bar.set_postfix(acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)))
        batch_bar.update()   
    batch_bar.close()
    print("Validation: {:.04f}%".format(100 * num_correct / len(val_dataset)))
    val_acc = 100 * num_correct / len(val_dataset)

    if val_acc > best_acc:
        torch.save(model.state_dict(), best_path + '/val_{:.03f}.pth'.format(val_acc))
        best_acc = val_acc
        print("----- best acc saved -----")


# Test
test_dataset = ClassificationTestSet(TEST_DIR, ttf.Compose(val_transforms))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         drop_last=False, num_workers=1)
model.eval()
batch_bar = tqdm(total=len(test_loader), dynamic_ncols=True, position=0, leave=False, desc='Test')
res = []
for i, (x) in enumerate(test_loader):

    x = x.cuda()
    with torch.no_grad():
        outputs = model(x)
        pred = torch.argmax(outputs, dim=1)
        res.extend(pred.tolist())

    batch_bar.update()    
batch_bar.close()

with open("/home/mmlab/idl/HW2P2/result/submission/submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(test_dataset)):
        f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))

