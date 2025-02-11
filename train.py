import os
import warnings
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from dataset import PartNetMemoryDataset, TrainPartiaPartNetDataset
from model import PointNetSegmentation
from partnet import CATEGORIES

warnings.filterwarnings("ignore")
torch.manual_seed(0) # 重み固定


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_names", nargs="*", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cospa_train", type=bool, default=False)
    parser.add_argument("--train_split", type=float, default=0.8)

    return parser.parse_args()


# lossを計算する引数を作成
def calc_pred_and_true_label(pred_y, batch_data, cospa_train):
    if not cospa_train:
        return pred_y, batch_data.y.detach().to(device)

    sum = 0
    pred_labels = []
    indices = []
    true_y = batch_data.y.detach().to(device)
    this_batch_size = batch_data.batch.detach().max() + 1
    # print("===============================================")
    for i in range(this_batch_size):
        o_start, o_end = batch_data.ptr[i], batch_data.ptr[i + 1]
        ano_start, ano_end = 0, int(batch_data.ano_num[i])
        if i != 0:
            ano_start = sum
            ano_end = sum + int(batch_data.ano_num[i])
        sum += int(batch_data.ano_num[i])

        annotations = batch_data.ano_x[ano_start:ano_end]
        origin = batch_data.x[o_start:o_end]
        for annotation in annotations:
            # print(annotation.shape, origin.shape)
            matches = torch.all(annotation == origin, dim=1)
            idx = torch.nonzero(matches, as_tuple=False)
            if idx.size(0) > 0:
                indices.append(int(batch_data.ptr[i] + idx.item()))
    # print(indices)
    pred_labels = pred_y[indices, :]
    # true_labels = true_y[indices]
    return pred_labels, batch_data.ano_y


args = get_arguments()


train_dataset = PartNetMemoryDataset(args.dataset_dir, args.target_names, train_split=args.train_split)
if args.cospa_train:
    train_dataset = TrainPartiaPartNetDataset(
        args.dataset_dir, args.annotation_csv, args.target_names, train_split=args.train_split
    )

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
        pred_y, true_y = calc_pred_and_true_label(pred_y, batch_data, args.cospa_train)
        pred_class = pred_y.argmax(dim=1)

        # if epoch % 10 == 0:
        #     labels = []
        #     for i in range(batch_data.ptr[2] - batch_data.ptr[1]):
        #         labels.append(
        #             [
        #                 batch_data.x[batch_data.ptr[1] + i][0].item(),
        #                 batch_data.x[batch_data.ptr[1] + i][1].item(),
        #                 batch_data.x[batch_data.ptr[1] + i][2].item(),
        #                 true_y[batch_data.ptr[1] + i].item(),
        #             ]
        #         )
        #     os.makedirs(f"{args.res_dir}/pred", exist_ok=True)
        #     torch.save(torch.tensor(labels), f"{args.res_dir}/pred/test_{epoch}.pth")
        # if epoch % 10 == 0:
        #     labels = []
        #     for i in range(batch_data.ptr[1] - 1):
        #         labels.append(
        #             [
        #                 batch_data.x[i][0].item(),
        #                 batch_data.x[i][1].item(),
        #                 batch_data.x[i][2].item(),
        #                 true_y[i].item(),
        #             ]
        #         )
        #     os.makedirs(f"{args.res_dir}/pred", exist_ok=True)
        #     torch.save(torch.tensor(labels), f"{args.res_dir}/pred/test_{epoch}.pth")

        class_loss = criteria(pred_y, true_y)
        accuracy = float((pred_class == true_y).sum()) / float(this_batch_size)

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
