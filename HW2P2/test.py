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

from mobilenetV2 import MobileNetV2
from collections import OrderedDict

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



model_name = 'val_74.011.pth'
model = MobileNetV2()
state_dict = torch.load("/home/mmlab/idl/HW2P2/result/model/{}".format(model_name))

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)

model.cuda()

DATA_DIR = "/home/mmlab/idl/HW2P2/hw2p2-data"
TEST_DIR = osp.join(DATA_DIR, "classification/classification/test")

batch_size = 64
val_transforms = [ttf.ToTensor()]
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

with open("/home/mmlab/idl/HW2P2/result/submission/submission_{}.csv".format(model_name.split('.')[0]), "w+") as f:
    f.write("id,label\n")
    for i in range(len(test_dataset)):
        f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", res[i]))