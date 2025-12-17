import os
import torch
from torch.utils.data import Dataset, DataLoader
from models import CAFB_PA
from train import MAHCD
import numpy as np
import cv2
from tqdm import tqdm 

data_dir = "/host/code/data/1mydataset/"
txt_path = "/host/code/data/1mydataset/list73_total_random/"

output_pre_path = ""
if not os.path.exists(output_pre_path):  
    os.makedirs(output_pre_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

checkpoint_path =""

 
model = CAFB_PA()
model = model.to(device)  
checkpoint = torch.load(checkpoint_path)  

model.load_state_dict(checkpoint['model_state_dict'])  

test_data = MAHCD(data_dir,txt_path, "val")
print("test",len(test_data))
test_dataloader = DataLoader(test_data, batch_size=1,
                                       shuffle=False, num_workers=4, pin_memory=True)

intersection_total = 0
union_total = 0
TP_total = 0
TN_total = 0
FP_total = 0
FN_total = 0
model.eval()
for i, data in enumerate(tqdm(test_dataloader, desc="test", total=len(test_dataloader))):  
    x1, x2, lbl,filename = data
    name = filename[0]
    x1 = x1.to(device, dtype=torch.float)
    x2 = x2.to(device, dtype=torch.float)
    lbl = lbl.to(device, dtype=torch.long)
    y = model(x1, x2)
    pre_label = y[:, 0] < y[:, 1]  # 第1维大的为真

    pre = pre_label.squeeze(0)
    pre = (pre*255).cpu().numpy().astype(np.uint8)

    cv2.imwrite(os.path.join(output_pre_path, name), pre)

    TP = intersection = pre_label[lbl == 1].long().sum()
    union = pre_label.sum() + lbl.sum() - intersection
    intersection_total += intersection.item()
    union_total += union.item()

    FN = (~pre_label)[lbl == 1].long().sum()
    TN = (~pre_label)[lbl == 0].long().sum()
    FP = pre_label[lbl == 0].long().sum()
    TP_total += TP.item()
    TN_total += TN.item()
    FP_total += FP.item()
    FN_total += FN.item()

lbl_total = FP_total + TP_total + TN_total + FN_total
precision = TP_total / (TP_total + FP_total + 0.01)
recall = TP_total / (TP_total + FN_total + 0.01)
F1 = 2 * precision * recall / (precision + recall + 0.01)
OA = (TP_total + TN_total) / (lbl_total)
iou = float(intersection_total) / (union_total + 0.01)
print("diff_lbl_sum:%d,precision:%.5f,recall:%.5f,F1 score:%.5f,OA:%.5f,iou:%.5f" % \
             (lbl_total - FP_total - TP_total - TN_total - FN_total, precision, recall, F1, OA, iou))
