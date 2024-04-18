import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from loguru import logger
from .utils import get_lr
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc


def fit_one_epoch(model_train, model, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda):
    total_loss      = 0
    total_accuracy  = 0
    val_accuracy = 0
    val_loss        = 0
    predictions = []  # 存储所有batch的预测概率
    targets_list = []  # 存储所有batch的真实标签
    tensorboard_save_path = 'logs/tensorboard_logs'
    if not os.path.exists(tensorboard_save_path):
        os.makedirs(tensorboard_save_path)
    model_train.train()
    logger.info('Start Train')
    writer = SummaryWriter(tensorboard_save_path)
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step: 
                break
            images, targets = batch
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor)  # [batch_size,3,224,224]
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()  # 类型[1,0] 长度和batch_size有关 代表每张照片属于什么类
                if cuda:
                    images = images.cuda()
                    targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model_train(images)  # output.shape=[batch_size,num_classes]
            # ----------------2024.4.16 增加ROC---------------------------
            num_classes = outputs.shape[1]
            prob = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()
            predictions.append(prob)
            targets_list.append(targets.cpu().numpy())
            # nn.CrossEntropyLoss()内部会自动加上sofmax层，只不过加的是log_softmax层
            loss_value = nn.CrossEntropyLoss()(outputs, targets)
            loss_value.backward()
            optimizer.step()

            total_loss += loss_value.item()
            with torch.no_grad():
                accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            writer.add_scalar('Train/acc', total_accuracy / (iteration + 1), epoch * epoch_step + iteration)
            writer.add_scalar('Train/Loss', total_loss / (iteration + 1), epoch * epoch_step + iteration)
            writer.add_scalar('Train/lr', get_lr(optimizer), epoch * epoch_step + iteration)

    predictions = np.concatenate(predictions)  # shape is [180,2]
    targets_list = np.concatenate(targets_list)  # shape is [180]
    # -----计算ROC-------
    roc_auc_scores = []
    # 存储每个类别的 fpr 和 tpr
    roc_curves = {}
    for class_idx in range(num_classes):
        # 将当前类别作为正样本
        positive_probs = predictions[:, class_idx]
        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(targets_list, positive_probs, pos_label=class_idx)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=1, label='ROC fold (area = %0.4f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'logs/roc_curve_class_{class_idx}.png')
        plt.close()

    logger.info('Finish Train')
    model_train.eval()
    logger.info('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch
            with torch.no_grad():
                images  = torch.from_numpy(images).type(torch.FloatTensor)
                targets = torch.from_numpy(targets).type(torch.FloatTensor).long()
                if cuda:
                    images  = images.cuda()
                    targets = targets.cuda()

                optimizer.zero_grad()

                outputs     = model_train(images)

                loss_value  = nn.CrossEntropyLoss()(outputs, targets)
                
                val_loss    += loss_value.item()
                with torch.no_grad():
                    accuracy = torch.mean(
                        (torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
                    val_accuracy += accuracy.item()
                
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'val_accuracy': val_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    logger.info('Finish Validation')
    logger.info('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    logger.info('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), 'logs/epoch%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
    writer.close()