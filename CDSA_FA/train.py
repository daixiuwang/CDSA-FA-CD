import os
import numpy as np
from tqdm import tqdm 
from utils import *
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from models.model import CAFB_PA
import logging
import sys
import math
import argparse
os.environ['TORCH_HOME'] = './weight/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/host/code/data/MAHCD/")
    parser.add_argument('--txt_path', type=str, default="/host/code/data/list73_total_ramdom")
    parser.add_argument('--resume_model', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.007, help='learning rate: HTCD lr=0.015  MAHCD lr=0.007 ')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch_lr_decay', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.8, help='decay rate of learning rate: HTCD 0.9  MAHCD 0.8 ')
    parser.add_argument('--weight_iou_loss', type=float, default=0.5)
    parser.add_argument('--weight_ce_loss', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    opt = parser.parse_args()

    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_dir + '/logging.log', level=logging.INFO,

                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
                        
    from tensorboardX import SummaryWriter

    my_log_info='training Mymodel with MAHCD dataset\nlogdir:'+log_dir
    writer = SummaryWriter(log_dir + '/TensorBoard')
    writer.add_text(tag='my_log_info',text_string=my_log_info)

    weights_dir=log_dir+'/weights'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    #打印到控制台
    logger=logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info(my_log_info)

    train_data = MAHCD(opt.data_dir,opt.txt_path, "train")
    validation_data = MAHCD(opt.data_dir,opt.txt_path, "val")
    logging.info('training set:%d patches' % len(train_data))
    logging.info('validation set:%d patches' % len(validation_data))
    
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=1,
                                       shuffle=False, num_workers=4, pin_memory=True)

    model = CAFB_PA()
    model.to(device, dtype=torch.float)
    #model = nn.DataParallel(model, device_ids=device_ids)

    start_epoch = 0
    if opt.resume_model!=None:
        checkpoint=torch.load(opt.esume_model)
        start_epoch = checkpoint['epoch']  
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info('resume success')

    parameters_tot = 0
    for nom, param in model.named_parameters():
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    logging.info("Number of model parameters %d",parameters_tot)

    lr_step_size = math.ceil(len(train_data) / opt.batch_size) * opt.epoch_lr_decay
    optimizer = SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_step_size,gamma=opt.lr_decay)

    loss_t=LossTotal(weight_ba_loss=opt.weight_iou_loss,weight_ce_loss=opt.weight_ce_loss,device= device)

    ave_loss_total=[]
    ave_loss_validation=[]
    ave_loss_100 = []
    logger.info("lr_scheduler: %s",lr_scheduler)
    logger.info('training ready.MetaData:\n lr:%f,lr_step_size:%d,lr_decay:%f,epoch_lr_decay:%f,momentum:%f,weight_decay:%f\n'
                'weight_ba_loss:%f,weight_ce_loss:%f\n'
                %(opt.lr,lr_step_size,opt.lr_decay,opt.epoch_lr_decay,opt.momentum,opt.weight_decay,opt.weight_iou_loss,opt.weight_ce_loss))

    for epoch in range(start_epoch+1, opt.epoch+1):
        for param_group in optimizer.param_groups:
            logger.info("lr----%f"%(param_group['lr']))
        loss_100=[]
        loss_total=[]
        model.train()
        union_total=0
        intersection_total=0
        for i, data in enumerate(tqdm(train_dataloader, desc="Training", total=len(train_dataloader))):
            x1, x2, lbl, _ = data
            x1 = x1.to(device, dtype=torch.float)
            x2 = x2.to(device, dtype=torch.float)
            lbl = lbl.to(device, dtype=torch.long)
            y = model(x1,x2)
            optimizer.zero_grad()
            loss = loss_t(y, lbl)
            loss.backward()
            loss_100.append(loss.item())
            loss_total.append(loss.item())
            optimizer.step()
            lr_scheduler.step()
            pre_label = y[:, 0] < y[:, 1]  # 第1维大的为 changed
            intersection = pre_label[lbl == 1].long().sum()
            union = pre_label.sum() + lbl.sum() - intersection
            intersection_total =intersection_total+ intersection
            union_total = union_total+union
            if(i%500==0 and i>0):
                mean_loss=np.mean(loss_100)
                writer.add_scalar('loss_100',mean_loss,global_step=len(ave_loss_100))
                logging.info('average loss of batch '+str(i-499)+'-'+str(i)+':'+str(mean_loss))
                ave_loss_100.append(mean_loss)
                loss_100 = []
                
                
        mean_loss=np.mean(loss_total)
        writer.add_scalar('loss_total',mean_loss,global_step=epoch)
        iou=(intersection_total.float()/union_total.float()).cpu().numpy()
        writer.add_scalar('iou_train',iou,global_step=epoch)
        logging.info('average loss of epoch'+str(epoch)+': ' +str(mean_loss))
        logging.info('average train iou of epoch'+ "-" + str(epoch) +': ' + str(iou))
        
        ave_loss_total.append(mean_loss)
        
        # validation
        if epoch % 10 ==0 or epoch > 180:
            loss_total = []
            intersection_total = 0
            union_total = 0
            TP_total = 0
            TN_total = 0
            FP_total = 0
            FN_total = 0
            model.eval()
            for i, data in enumerate(tqdm(validation_dataloader, desc="val", total=len(validation_dataloader))):
                x1, x2, lbl, _ = data
                x1 = x1.to(device, dtype=torch.float)
                x2 = x2.to(device, dtype=torch.float)
                lbl = lbl.to(device, dtype=torch.long)
                y = model(x1, x2)
                loss = loss_t(y, lbl)
                loss_total.append(loss.item())
                pre_label = y[:, 0] < y[:, 1]  # 第1维大的为真
                TP = intersection = pre_label[lbl == 1].long().sum()
                union = pre_label.sum() + lbl.sum() - intersection
                intersection_total = intersection_total+intersection.item()
                union_total =union_total + union.item()
                FN = (~pre_label)[lbl == 1].long().sum()
                TN = (~pre_label)[lbl == 0].long().sum()
                FP = pre_label[lbl == 0].long().sum()
                TP_total =TP_total+ TP.item()
                TN_total =TN_total + TN.item()
                FP_total =FP_total +FP.item()
                FN_total =FN_total +FN.item()

            mean_loss = np.mean(loss_total)
            writer.add_scalar('loss_validation', mean_loss, global_step=epoch)

            lbl_total = FP_total + TP_total + TN_total + FN_total
            precision = TP_total / (TP_total + FP_total+0.01)
            recall = TP_total / (TP_total + FN_total+0.01)
            F1 = 2 * precision * recall / (precision + recall+0.01)
            OA = (TP_total + TN_total) / (lbl_total)
            iou = float(intersection_total) / (union_total+0.01)

            metric_msg = "diff_lbl_sum:%d,precision:%.5f,recall:%.5f,F1 score:%.5f,OA:%.5f,iou:%.5f" % \
                         (lbl_total - FP_total - TP_total - TN_total - FN_total, precision, recall, F1, OA, iou)
            logging.info('validation metrics of epoch'+"-" + str(epoch)+" " + metric_msg)

            ave_loss_validation.append(mean_loss)
            logging.info('validation loss:' + str(mean_loss))

            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, weights_dir + '/model_para_{}.pth'.format(epoch))

        
if __name__ == '__main__':
    main()