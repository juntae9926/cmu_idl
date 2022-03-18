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
import pdb

from mobilenetV2 import MobileNetV2

class VerificationDataset(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in data_dir
        self.img_paths = list(map(lambda fname: osp.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # We return the image, as well as the path to that image (relative path)
        return self.transforms(Image.open(self.img_paths[idx])), osp.relpath(self.img_paths[idx], self.data_dir)

if __name__ == "__main__":
    batch_size = 1
    DATA_DIR = "/home/mmlab/idl/HW2P2/hw2p2-data"
    val_transforms = [ttf.ToTensor()]
    val_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/test"), ttf.Compose(val_transforms))
    val_ver_loader = torch.utils.data.DataLoader(val_veri_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_veri_dataset = VerificationDataset(osp.join(DATA_DIR, "verification/verification/test"), ttf.Compose(val_transforms))
    test_ver_loader = torch.utils.data.DataLoader(test_veri_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    model = MobileNetV2()
    model.load_state_dict(torch.load("/home/mmlab/idl/HW2P2/result/model/val_73.566.pth"))
    model.eval()
    #model.cuda()

    feats_dict = dict()

    for batch_idx, (imgs, path_names) in tqdm(enumerate(test_ver_loader), total=len(test_ver_loader), position=0, leave=False, desc='Val'):
        #imgs = imgs.cuda()

        with torch.no_grad():
            # Note that we return the feats here, not the final outputs
            # Feel free to try the final outputs too!
            feats = model(imgs, return_feats=True) 
        
        # TODO: Now we have features and the image path names. What to do with them?
        # Hint: use the feats_dict somehow.
        for i in range(len(imgs)):
            feats_dict[path_names[i]] = feats[i]

    print(len(feats_dict))


    # What does this dict look like?
    #print(list(feats_dict.items())[0])

    # We use cosine similarity between feature embeddings.
    # TODO: Find the relevant function in pytorch and read its documentation.
    similarity_metric = nn.CosineSimilarity(dim=0, eps=1e-08)

    #val_veri_csv = osp.join(DATA_DIR, "verification/verification/verification_test.csv")
    test_veri_csv = osp.join(DATA_DIR, "verification/verification/verification_test.csv")


    # Now, loop through the csv and compare each pair, getting the similarity between them
    pred_similarities = []
    gt_similarities = []
    for line in tqdm(open(test_veri_csv).read().splitlines()[1:], position=0, leave=False): # skip header
        img_path1, img_path2 = line.split(",")
        #img_path1, img_path2, gt = line.split(",")

        feats_1 = feats_dict[img_path1.split('/')[1]]
        feats_2 = feats_dict[img_path2.split('/')[1]]
        similarity = similarity_metric(feats_1, feats_2)
        pred_similarities.append(similarity.item())
        #gt_similarities.append(int(gt))

    pred_similarities = np.array(pred_similarities)
    #gt_similarities = np.array(gt_similarities)

    #print("AUC:", roc_auc_score(gt_similarities, pred_similarities))


    with open("/home/mmlab/idl/HW2P2/verification.csv", "w+") as f:
        f.write("id,match\n")
        for i in range(len(pred_similarities)):
            f.write("{},{}\n".format(i, pred_similarities[i]))
    print("End")

