import os
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PartNet2MemoryDataset
from model import PointNet2Segmentation
from partnet import CATEGORIES

warnings.filterwarnings("ignore")
torch.manual_seed(0) # 重み固定

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def to_categorical2(B, num_classes):
    tensor = torch.ones((B, num_classes))
    return tensor.to(device)

def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, required=True)
    # parser.add_argument("--point_dir", type=str, required=True)
    # parser.add_argument("--label_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_names", nargs="*", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)

    return parser.parse_args()


args = get_arguments()

# train_dataset = PartNet2MemoryDataset(args.point_dir, args.label_dir)
train_dataset = PartNet2MemoryDataset(args.dataset_dir, args.target_names)

device = (
    torch.device(f"cuda:{args.gpu}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
output_dim = 0
for target_name in args.target_names:
    output_dim += int(CATEGORIES[target_name]["part_num"])
# model = PointNet2Segmentation(output_dim)
# model = model.to(device)
from models import pointnet2_part_seg_msg

model = pointnet2_part_seg_msg.get_model(output_dim).to(device)

optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=args.epoch // 4, gamma=0.5
)

log_dir = f"{args.res_dir}/log"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

criteria = torch.nn.CrossEntropyLoss()

from tqdm import tqdm

for epoch in tqdm(range(args.epoch)):
    model = model.train()

    losses = []
    for points, labels in train_dataloader:
        points = points.to(device)
        labels = labels.to(device)
        this_batch_size = labels.size()[0]

        pred_y, _ = model(points, to_categorical2(this_batch_size, 1))

        # print(seg_pred.shape, labels.shape)

        # pred_y = seg_pred.contiguous().view(-1, output_dim)
        # print(pred_y.shape)
        # print(pred_y)
        pred_y = pred_y.reshape(-1, pred_y.size(-1))  # (バッチサイズ * 点数, クラス数)
        true_y = labels.reshape(-1)

        class_loss = criteria(pred_y, true_y)
        accuracy = float((pred_y.argmax(dim=1) == true_y).sum()) / float(
            this_batch_size
        )

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

        loss = class_loss

        losses.append(
            {
                "loss": loss.item(),
                "class_loss": class_loss.item(),
                # "reg_loss": reg_loss.item(),
                "accuracy": accuracy,
                "seen": float(this_batch_size),
            }
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    if epoch % 50 == 0:
        model_path = f"{log_dir}/model_{epoch:06}.pth"
        torch.save(model.state_dict(), model_path)

    loss = 0
    class_loss = 0
    accuracy = 0
    seen = 0
    for d in losses:
        seen = seen + d["seen"]
        loss = loss + d["loss"] * d["seen"]
        class_loss = class_loss + d["class_loss"] * d["seen"]
        accuracy = accuracy + d["accuracy"] * d["seen"]
    loss = loss / seen
    class_loss = class_loss / seen
    accuracy = accuracy / seen
    writer.add_scalar("train_epoch/loss", loss, epoch)
    writer.add_scalar("train_epoch/class_loss", class_loss, epoch)
    writer.add_scalar("train_epoch/accuracy", accuracy, epoch)

torch.save(model.state_dict(), f"{args.res_dir}/model.pth")
