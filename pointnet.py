import os
import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from dataset import PartNetMemoryDataset
from model import PointNetSegmentation
from partnet import CATEGORIES

warnings.filterwarnings("ignore")
torch.manual_seed(0) # 重み固定


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--train_split", type=float, default=0.8)

    return parser.parse_args()


args = get_arguments()

# train_dataset = PartNetMemoryDataset(args.point_dir, args.label_dir)
train_dataset = PartNetMemoryDataset(args.dataset_dir, args.target_names, train_split=args.train_split)

device = (
    torch.device(f"cuda:{args.gpu}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)
output_dim = 0
for target_name in args.target_names:
    output_dim += int(CATEGORIES[target_name]["part_num"])
model = PointNetSegmentation(output_dim)
model = model.to(device)

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
    for batch_data in train_dataloader:
        batch_data = batch_data.to(device)
        this_batch_size = batch_data.batch.detach().max() + 1

        pred_y, _, feature_transform = model(batch_data)
        true_y = batch_data.y.detach().to(device)

        class_loss = criteria(pred_y, true_y)
        accuracy = float((pred_y.argmax(dim=1) == true_y).sum()) / float(
            this_batch_size
        )

        id_matrix = (
            torch.eye(feature_transform.shape[1])
            .to(feature_transform.device)
            .view(1, 64, 64)
            .repeat(feature_transform.shape[0], 1, 1)
        )
        transform_norm = torch.norm(
            torch.bmm(feature_transform, feature_transform.transpose(1, 2)) - id_matrix,
            dim=(1, 2),
        )
        reg_loss = transform_norm.mean()

        loss = class_loss + reg_loss * 0.001

        losses.append(
            {
                "loss": loss.item(),
                "class_loss": class_loss.item(),
                "reg_loss": reg_loss.item(),
                "accuracy": accuracy,
                "seen": float(this_batch_size),
            }
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    if epoch % 10 == 0:
        model_path = f"{log_dir}/model_{epoch:06}.pth"
        torch.save(model.state_dict(), model_path)

    loss = 0
    class_loss = 0
    reg_loss = 0
    accuracy = 0
    seen = 0
    for d in losses:
        seen = seen + d["seen"]
        loss = loss + d["loss"] * d["seen"]
        class_loss = class_loss + d["class_loss"] * d["seen"]
        reg_loss = reg_loss + d["reg_loss"] * d["seen"]
        accuracy = accuracy + d["accuracy"] * d["seen"]
    loss = loss / seen
    class_loss = class_loss / seen
    reg_loss = reg_loss / seen
    accuracy = accuracy / seen
    writer.add_scalar("train_epoch/loss", loss, epoch)
    writer.add_scalar("train_epoch/class_loss", class_loss, epoch)
    writer.add_scalar("train_epoch/reg_loss", reg_loss, epoch)
    writer.add_scalar("train_epoch/accuracy", accuracy, epoch)

torch.save(model.state_dict(), f"{args.res_dir}/model.pth")
