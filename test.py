import os
import ssl
import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import TestPartNetMemoryDataset
from model import PointNetSegmentation
from partnet import CATEGORIES

warnings.filterwarnings("ignore")


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument("--point_dir", type=str, required=True)
    # parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--target_names", nargs="*", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_split", type=float, default=0.8)

    return parser.parse_args()


args = get_arguments()

if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"model path not found: {args.model_path}")

test_dataset = TestPartNetMemoryDataset(args.dataset_dir, args.target_names, train_split=args.train_split)
# test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

log_dir = f"{args.res_dir}/log_test"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

device = (
    torch.device(f"cuda:{args.gpu}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
output_dim = 0
for target_name in args.target_names:
    output_dim += int(CATEGORIES[target_name]["part_num"])
print(f"output_dim: {output_dim}")
model = PointNetSegmentation(output_dim)
model = model.to(device)
model.load_state_dict(torch.load(args.model_path))

criteria = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    model = model.eval()

    losses = []
    num = 0
    mIoUs = []
    iou_dict = {}

    for i in range(output_dim):
        iou_dict[i] = []

    for batch_data in tqdm(test_dataloader, total=len(test_dataloader)):
        labels = []
        batch_data = batch_data.to(device)
        this_batch_size = batch_data.batch.detach().max() + 1

        pred_y, _, _ = model(batch_data)
        true_y = batch_data.y.detach().to(device)

        class_loss = criteria(pred_y, true_y)
        pred_class = pred_y.argmax(dim=1)
        correct_predictions = torch.eq(pred_class, true_y)
        num_correct = correct_predictions.sum().item()
        num_total = true_y.size(0)

        accuracy = num_correct / num_total

        # print(f"Accuracy: {accuracy * 100:.2f}%")
        for i in range(len(batch_data.x)):
            labels.append(
                [
                    batch_data.x[i][0].item(),
                    batch_data.x[i][1].item(),
                    batch_data.x[i][2].item(),
                    pred_class[i].item(),
                ]
            )

        # 評価指標計算
        # accuracy = float((pred_class == true_y).sum()) / float(
        #     this_batch_size
        # )

        ## mIoU計算
        class_num = output_dim
        ious = []
        for cls in range(class_num):
            pred_mask = pred_class == cls
            true_mask = true_y == cls
            intersection = (pred_mask & true_mask).sum().item()
            union = (pred_mask | true_mask).sum().item()
            if union == 0:
                iou = 1.0
            else:
                iou = intersection / union

            ious.append(iou)
            iou_dict[cls].append(iou)
            # print(f"IoU_{cls}: {iou * 100:.2f}%")
        mIoU = sum(ious) / class_num
        # print(f"mIoU: {mIoU * 100:.2f}%")
        mIoUs.append(mIoU)

        # id_matrix = (
        #     torch.eye(feature_transform.shape[1])
        #     .to(feature_transform.device)
        #     .view(1, 64, 64)
        #     .repeat(feature_transform.shape[0], 1, 1)
        # )
        # transform_norm = torch.norm(
        #     torch.bmm(feature_transform, feature_transform.transpose(1, 2)) - id_matrix,
        #     dim=(1, 2),
        # )
        # reg_loss = transform_norm.mean()

        # loss = class_loss + reg_loss * 0.001 * 0.001

        # losses.append(
        #     {
        #         "loss": loss.item(),
        #         "class_loss": class_loss.item(),
        #         "reg_loss": reg_loss.item(),
        #         "accuracy": accuracy,
        #         "seen": float(this_batch_size),
        #     }
        # )

        os.makedirs(f"{args.res_dir}/labels", exist_ok=True)
        torch.save(torch.tensor(labels), f"{args.res_dir}/labels/labels_{num}.pth")
        num += 1

    # 平均と標準偏差を計算
    base_mIoUs = mIoUs
    mIoUs = torch.tensor(mIoUs)
    mean_mIoU = mIoUs.mean().item()
    std_mIoU = mIoUs.std().item()
    print(f"{mean_mIoU:.3f}±{std_mIoU:.3f}%")

    for key in iou_dict:
        iou = torch.tensor(iou_dict[key])
        mean_IoU = iou.mean().item()
        std_IoU = iou.std().item()
        print(f"{mean_IoU:.3f}±{std_IoU:.3f}%")

    # loss = 0
    # class_loss = 0
    # reg_loss = 0
    # accuracy = 0
    # seen = 0
    # for d in losses:
    #     seen = seen + d["seen"]
    #     loss = loss + d["loss"] * d["seen"]
    #     class_loss = class_loss + d["class_loss"] * d["seen"]
    #     reg_loss = reg_loss + d["reg_loss"] * d["seen"]
    #     accuracy = accuracy + d["accuracy"] * d["seen"]
    # loss = loss / seen
    # class_loss = class_loss / seen
    # reg_loss = reg_loss / seen
    # accuracy = accuracy / seen
    # writer.add_scalar("test_epoch/loss", loss, epoch)
    # writer.add_scalar("test_epoch/class_loss", class_loss, epoch)
    # writer.add_scalar("test_epoch/reg_loss", reg_loss, epoch)
    # writer.add_scalar("test_epoch/accuracy", accuracy, epoch)
