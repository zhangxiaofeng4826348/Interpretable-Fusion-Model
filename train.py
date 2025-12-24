import torch
import torch.nn as nn
from pandas.core.computation.expr import intersection
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CustomDataset
from model import UNet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from  Weighted_Focal_Loss import WeightedFocalLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DiceLoss(nn.Module):
    def __init__(self, beta=1):
        super(DiceLoss, self).__init__()
        self.beta = beta

    def forward(self, predicted, target):
        smooth = 1e-5
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1.0 - dice_coeff
        return loss

def dice_coefficient(predicted, target):
    smooth = 1e-5
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return dice_coeff

def calculate_iou(predicted, target):
    smooth = 1e-5
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def train_net(train_image_folder, train_mask_folder, val_image_folder,val_mask_folder,label_files):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = CustomDataset(train_image_folder, train_mask_folder, label_files, transform)
    val_dataset = CustomDataset(val_image_folder, val_mask_folder, label_files, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_epoch = 100
    batch_size = 12
    alpha = 0.1

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=all_epoch)

    num_malignant = 484
    num_benign = 210

    #分割和分类的损失函数
    criterion_seg = DiceLoss()
    # criterion_classification = nn.CrossEntropyLoss()
    criterion_classification = WeightedFocalLoss(num_malignant, num_benign, gamma=2.0)

    best_val_loss = float('inf')
    best_val_model_state = None

    best_train_loss = float('inf')
    best_train_model_state = None

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    train_dice_scores = []
    val_dice_scores = []

    train_ious = []
    val_ious = []

    lr = []

    for epoch in range(all_epoch):

        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_dice_score = 0.0
        train_iou = 0.0
        train_num_batches = len(train_dataloader)

        all_train_predicted_labels = []
        all_train_true_labels = []

        for batch in train_dataloader:
            images = batch['image'].to(device)
            train_labels = batch['label'].to(device)
            masks = batch['mask'].to(device)

            seg_outputs, outputs = model(images)
            optimizer.zero_grad()

            # 计算交叉熵损失
            classification_loss = criterion_classification(outputs, train_labels)

            seg_loss = criterion_seg(seg_outputs, masks)

            loss = alpha * classification_loss + (1 - alpha) * seg_loss

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            # 分割指标
            train_dice_score += dice_coefficient(seg_outputs, masks).item()
            train_iou += calculate_iou(seg_outputs, masks).item()

            # 分类指标
            _, predicted = torch.max(outputs, 1)
            total_train += train_labels.size(0)
            correct_train += (predicted == train_labels).sum().item()

            all_train_predicted_labels.extend(predicted.cpu().numpy())
            all_train_true_labels.extend(train_labels.cpu().numpy())

        train_dice_scores.append(train_dice_score / train_num_batches)
        train_ious.append(train_iou / train_num_batches)

        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_dataloader))
        train_accuracies.append(train_accuracy)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_dice_score = 0.0
        val_iou = 0.0
        val_num_batches = len(val_dataloader)

        all_val_predicted_labels = []
        all_val_true_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                val_images = batch['image'].to(device)
                val_labels = batch['label'].to(device)
                masks = batch['mask'].to(device)

                seg_outputs, outputs = model(val_images)

                val_calssification_loss = criterion_classification(outputs, val_labels)

                val_seg_loss = criterion_seg(seg_outputs, masks)

                loss = (1 - alpha) * val_seg_loss + alpha * val_calssification_loss

                val_loss += loss.item()

                val_dice_score += dice_coefficient(seg_outputs, masks).item()
                val_iou += calculate_iou(seg_outputs, masks).item()

                _, predicted = torch.max(outputs, 1)
                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()

                all_val_predicted_labels.extend(predicted.cpu().numpy())
                all_val_true_labels.extend(val_labels.cpu().numpy())

        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / len(val_dataloader))
        val_accuracies.append(val_accuracy)

        val_dice_scores.append(val_dice_score / val_num_batches)
        val_ious.append(val_iou / val_num_batches)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_model_state = model.state_dict()

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_model_state = model.state_dict()

        torch.save(model.state_dict(), f"model/epoch{epoch + 1}.pth")

        print(f'Epoch: {epoch+1}, '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Val Loss: {val_losses[-1]:.4f}, '
              f'Train Accuracy: {train_accuracies[-1]:.4f}, '
              f'Val Accuracy: {val_accuracies[-1]:.4f},'
              f'Train Dice Score: {train_dice_scores[-1]:.4f}, '
              f'Val Dice Score: {val_dice_scores[-1]:.4f},'
              f'Train IOU: {train_ious[-1]:.4f},'
              f'Val IOU: {val_ious[-1]:.4f}'
              )

        scheduler.step()
        lr.append(scheduler.get_lr()[0])

    fold_result_path = f"out/result/result.xlsx"

    df = pd.DataFrame({'Train Loss': train_losses, 'Val Loss': val_losses,
                       'Train Accuracy': train_accuracies, 'Val Accuracy': val_accuracies,
                       'Train Dice': train_dice_scores, 'Val Dice': val_dice_scores,
                       'Train IOU': train_ious, 'Val IOU': val_ious})
    df.to_excel(fold_result_path, index=False)

    torch.save(best_val_model_state,
               f"out/result/best_val_model.pth")

    torch.save(best_train_model_state,
               f"out/result/best_train_model.pth")

    print(f'training completed.')

    plt.plot(np.arange(len(lr)), lr)
    plt.savefig(
        f'out/picture/lr.png')
    plt.show()
    plt.close()

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Over Epochs')
    plt.legend()
    plt.savefig(
        f'out/picture/loss.png')
    plt.show()
    plt.close()

    plt.plot(train_dice_scores, label='Train Dice Score')
    plt.plot(val_dice_scores, label='Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title(f'Dice Score Over Epochs')
    plt.legend()
    plt.savefig(
        f'out/picture/dice_score.png')
    plt.show()
    plt.close()

    plt.plot(train_ious, label='Train IOU')
    plt.plot(val_ious, label='Validation IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.title(f'IOU Over Epochs')
    plt.legend()
    plt.savefig(
        f'out/picture/IOU.png')
    plt.show()
    plt.close()

    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Over Epochs')
    plt.legend()
    plt.savefig(f'out/picture/accuracy.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    train_image_folder = "data/data_e/train_image"
    train_mask_folder = "data/data_e/train_mask"
    val_image_folder = "data/val/val_image"
    val_mask_folder = "data/val/val_mask"
    label_files = "data/data_e/label_enhance.xlsx"

    train_net(train_image_folder, train_mask_folder, val_image_folder,val_mask_folder,label_files)

