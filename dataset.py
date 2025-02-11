import glob

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from partnet import CATEGORIES


class PartNetMemoryDataset(InMemoryDataset):
    def __init__(self, dataset_dir, target_names, transform=None, train_split=0.8):
        super(PartNetMemoryDataset, self).__init__(dataset_dir, target_names)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.train_split = train_split
        self.data, self.slices = self.costom_process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Rawファイルのダウンロードコード
        pass

    def costom_process(self):
        data_list = []
        for k in range(len(self.target_names)):
            target = CATEGORIES[self.target_names[k]]["id"]
            label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
            label_paths = natsorted(glob.glob(label_dir))
            boundary_index = int(len(label_paths) * self.train_split)
            for p in label_paths[:boundary_index]:
                label_file_name = p.split("/")[-1].split(".")[0]

                labels = []
                points = []
                with open(p, "r") as file:
                    label_lines = file.readlines()
                    for line in label_lines:
                        labels.append(int(line) - 1 + k * 2)

                point_dir = f"{self.dataset_dir}/{target}/points"
                with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                    point_lines = file.readlines()
                    for line in point_lines:
                        points.append(list(map(float, line.split(" "))))

                data = Data(
                    x=torch.tensor(points, dtype=torch.float),
                    y=torch.tensor(labels, dtype=torch.long),
                )
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices

# self.train_split = 0.8
from torch.utils.data import DataLoader as TDataLoader
from torch.utils.data import Dataset


def farthest_point_sample(point, label, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    point = np.array(point)
    label = np.array(label)
    N, D = np.array(point).shape
    if N < npoint:
        return [], [], False
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    label = label[centroids.astype(np.int32)]
    return point, label, True


class PartNet2MemoryDataset(Dataset):
    def __init__(self, dataset_dir, target_names, transform=None, train_split=0.8):
        # super(PartNet2MemoryDataset, self).__init__(dataset_dir, target_names)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.train_split = train_split
        self.points, self.labels = self.costom_process()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index: int):
        # print(self.points[index])
        return self.points[index], self.labels[index]

    def costom_process(self):
        points = []
        labels = []
        sampling_num = 2400
        for k in range(len(self.target_names)):
            target = CATEGORIES[self.target_names[k]]["id"]
            if target == "03797390":
                sampling_num = 2800
            label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
            label_paths = natsorted(glob.glob(label_dir))
            boundary_index = int(len(label_paths) * self.train_split)
            for p in tqdm(label_paths[:boundary_index]):
                label_file_name = p.split("/")[-1].split(".")[0]

                label = []
                point = []
                with open(p, "r") as file:
                    label_lines = file.readlines()
                    for line in label_lines:
                        label.append(int(line) - 1 + k * 2)

                point_dir = f"{self.dataset_dir}/{target}/points"
                with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                    point_lines = file.readlines()
                    for line in point_lines:
                        point.append(list(map(float, line.split(" "))))

                point, label, ok = farthest_point_sample(point, label, sampling_num)
                if not ok:
                    continue
                points.append(point)
                labels.append(label)

        return torch.tensor(points, dtype=torch.float).permute(0, 2, 1), torch.tensor(labels, dtype=torch.long)

class TestPartNet2MemoryDataset(Dataset):
    def __init__(self, dataset_dir, target_names, transform=None, train_split=0.5):
        # super(PartNet2MemoryDataset, self).__init__(dataset_dir, target_names)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.train_split = train_split
        self.points, self.labels = self.costom_process()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index: int):
        # print(self.points[index])
        return self.points[index], self.labels[index]

    def costom_process(self):
        points = []
        labels = []
        sampling_num = 2400
        for k in range(len(self.target_names)):
            target = CATEGORIES[self.target_names[k]]["id"]
            if target == "03797390":
                sampling_num = 2800
            label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
            label_paths = natsorted(glob.glob(label_dir))
            boundary_index = int(len(label_paths) * self.train_split)
            target_label_paths = label_paths[boundary_index:]
            if target == "02691156":
                target_label_paths = label_paths[boundary_index:boundary_index*2]
            for p in tqdm(target_label_paths):
                label_file_name = p.split("/")[-1].split(".")[0]

                label = []
                point = []
                with open(p, "r") as file:
                    label_lines = file.readlines()
                    for line in label_lines:
                        label.append(int(line) - 1 + k * 2)

                point_dir = f"{self.dataset_dir}/{target}/points"
                with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                    point_lines = file.readlines()
                    for line in point_lines:
                        point.append(list(map(float, line.split(" "))))

                point, label, ok = farthest_point_sample(point, label, sampling_num)
                if not ok:
                    continue
                points.append(point)
                labels.append(label)

        return torch.tensor(points, dtype=torch.float).permute(0, 2, 1), torch.tensor(labels, dtype=torch.long)

class Train2PartiaPartNetDataset(InMemoryDataset):
    def __init__(self, dataset_dir, target_names, annotation_csv, transform=None, train_split=0.8):
        # super(PartNet2MemoryDataset, self).__init__(dataset_dir, target_names)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.train_split = train_split
        self.load_point_anntation_csv(annotation_csv)
        self.points, self.labels, self.ids = self.costom_process()

    def load_point_anntation_csv(self, csv_file):
        print(csv_file)
        self.annotation_points = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index: int):
        id = self.ids[index]
        target = self.annotation_points[self.annotation_points["id"] == id]
        ano_x_values = target[["x", "y", "z"]].to_numpy().tolist()
        ano_y_values = target["label"].to_list()
        # print(self.points[index])
        return self.points[index], self.labels[index], torch.tensor(ano_x_values, dtype=torch.float).permute(1, 0), torch.tensor(ano_y_values, dtype=torch.long)

    def costom_process(self):
        points = []
        labels = []
        ids = []
        sampling_num = 2400
        for k in range(len(self.target_names)):
            target = CATEGORIES[self.target_names[k]]["id"]
            if target == "03797390":
                sampling_num = 2800

            label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
            label_paths = natsorted(glob.glob(label_dir))
            boundary_index = int(len(label_paths) * self.train_split)
            for p in tqdm(label_paths[:boundary_index]):
                label_file_name = p.split("/")[-1].split(".")[0]

                label = []
                point = []
                with open(p, "r") as file:
                    label_lines = file.readlines()
                    for line in label_lines:
                        label.append(int(line) - 1 + k * 2)

                point_dir = f"{self.dataset_dir}/{target}/points"
                with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                    point_lines = file.readlines()
                    for line in point_lines:
                        point.append(list(map(float, line.split(" "))))

                point, label, ok = farthest_point_sample(point, label, sampling_num)
                if not ok:
                    continue
                points.append(point)
                labels.append(label)
                ids.append(label_file_name)

        return torch.tensor(points, dtype=torch.float).permute(0, 2, 1), torch.tensor(labels, dtype=torch.long), ids


class TestPartNetMemoryDataset(InMemoryDataset):
    def __init__(self, dataset_dir, target_names, transform=None, train_split=0.8):
        super(TestPartNetMemoryDataset, self).__init__(dataset_dir, target_names)
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.train_split = train_split
        self.data, self.slices = self.custom_process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def custom_process(self):
        data_list = []
        for k in range(len(self.target_names)):
            target = CATEGORIES[self.target_names[k]]["id"]
            label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
            label_paths = natsorted(glob.glob(label_dir))
            boundary_index = int(len(label_paths) * self.train_split)
            target_label_paths = label_paths[boundary_index:]
            if target == "02691156":
                target_label_paths = label_paths[boundary_index:boundary_index*2]
            for p in target_label_paths:
                label_file_name = p.split("/")[-1].split(".")[0]

                labels = []
                points = []
                with open(p, "r") as file:
                    label_lines = file.readlines()
                    for line in label_lines:
                        labels.append(int(line) - 1 + k * 2)

                point_dir = f"{self.dataset_dir}/{target}/points"
                with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                    point_lines = file.readlines()
                    for line in point_lines:
                        points.append(list(map(float, line.split(" "))))

                data = Data(
                    x=torch.tensor(points, dtype=torch.float),
                    y=torch.tensor(labels, dtype=torch.long),
                )
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices


# カスタムデータセットクラスの定義
class TrainPartiaPartNetDataset(InMemoryDataset):
    def __init__(self, dataset_dir, annotation_csv, target_names, train_split=0.8):
        super(TrainPartiaPartNetDataset, self).__init__(
            dataset_dir, annotation_csv, target_names
        )
        self.dataset_dir = dataset_dir
        self.target_names = target_names
        self.train_split = train_split
        self.load_point_anntation_csv(annotation_csv)
        self._data, self.slices = self.custom_prosess()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def load_point_anntation_csv(self, csv_file):
        self.annotation_points = pd.read_csv(csv_file)

    def custom_prosess(self):
        data_list = []
        for k in range(len(self.target_names)):
            target = CATEGORIES[self.target_names[k]]["id"]
            label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
            label_paths = natsorted(glob.glob(label_dir))
            boundary_index = int(len(label_paths) * self.train_split)
            for p in label_paths[:boundary_index]:
                label_file_name = p.split("/")[-1].split(".")[0]

                labels = []
                points = []
                with open(p, "r") as file:
                    label_lines = file.readlines()
                    for line in label_lines:
                        labels.append(int(line) - 1 + k * 2)

                point_dir = f"{self.dataset_dir}/{target}/points"
                with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                    point_lines = file.readlines()
                    for line in point_lines:
                        points.append(list(map(float, line.split(" "))))

                data = Data(
                    id=label_file_name,
                    pos=torch.tensor(points, dtype=torch.float),
                    y=torch.tensor(labels, dtype=torch.long),
                )
                data_list.append(data)

        data, slices = self.collate(data_list)
        return data, slices

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = self.get(idx)
            target = self.annotation_points[self.annotation_points["id"] == data.id]
            ano_x_values = target[["x", "y", "z"]].to_numpy().tolist()
            ano_y_values = target["label"].to_list()
            data = Data(
                id=data.id,
                x=data.pos,
                y=data.y,
                ano_x=torch.tensor(ano_x_values, dtype=torch.float),
                ano_y=torch.tensor(ano_y_values, dtype=torch.long),
                ano_num=len(ano_x_values),
            )

            return data

# from torch.utils.data import Dataset


# class PartNet2MemoryDataset(Dataset):
#     def __init__(self, dataset_dir, target_names, transform=None, self.train_split=0.8):
#         super(PartNet2MemoryDataset, self).__init__(dataset_dir, target_names)
#         self.transform = transform
#         self.dataset_dir = dataset_dir
#         self.target_names = target_names
#         self.data, self.slices = self.costom_process()

#     @property
#     def raw_file_names(self):
#         return []

#     @property
#     def processed_file_names(self):
#         return []

#     def download(self):
#         # Rawファイルのダウンロードコード
#         pass

#     # def __getitem__(self, index):


#     def costom_process(self):
#         data_list = []
#         for k in range(len(self.target_names)):
#             target = CATEGORIES[self.target_names[k]]["id"]
#             label_dir = f"{self.dataset_dir}/{target}/expert_verified/points_label/*"
#             label_paths = natsorted(glob.glob(label_dir))
#             boundary_index = int(len(label_paths) * self.train_split)
#             for p in label_paths[:boundary_index]:
#                 label_file_name = p.split("/")[-1].split(".")[0]

#                 labels = []
#                 points = []
#                 with open(p, "r") as file:
#                     label_lines = file.readlines()
#                     for line in label_lines:
#                         labels.append(int(line) - 1 + k * 2)

#                 point_dir = f"{self.dataset_dir}/{target}/points"
#                 with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
#                     point_lines = file.readlines()
#                     for line in point_lines:
#                         points.append(list(map(float, line.split(" "))))
#                 print("num_points:", len(points))

#                 data = Data(
#                     x=torch.tensor(points, dtype=torch.float),
#                     y=torch.tensor(labels, dtype=torch.long),
#                 )
#                 data_list.append(data)

#         data, slices = self.collate(data_list)
#         return data, slices


# # # データセットの使用例
# dataset = TrainPartiaPartNetDataset(
#     dataset_dir="/data/Users/watanabe/GlobalPointCoSPA/data/PartAnnotation",
#     annotation_csv="/data/Users/watanabe/GlobalPointCoSPA/csv/Cap/2/1.csv",
#     target_names=["Cap"],
# )
# # # dataset = PartNetMemoryDataset(root="/tmp/MyInMemoryDataset")
# loader = DataLoader(dataset, batch_size=8, shuffle=True)
# batch = next(iter(loader))
# print(batch)

# dataset = Train2PartiaPartNetDataset(
#     dataset_dir="/data/Users/watanabe/M2/GlobalPointCoSPA/data",
#     target_names=["Mug"],
#     annotation_csv="/data/Users/watanabe/M2/GlobalPointCoSPA/csv/simple/Mug/0/0.01.csv",
#     train_split=0.5,
# )
# loader = TDataLoader(dataset, batch_size=8, shuffle=True)
# batch = next(iter(loader))
# print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
# sum = 0
# for i in range(8):
#     print(batch.ptr[i], batch.ptr[i + 1])
#     o_start, o_end = batch.ptr[i], batch.ptr[i + 1]
#     ano_start, ano_end = 0, batch.ano_num[i]
#     if i != 0:
#         ano_start = sum
#         ano_end = sum + batch.ano_num[i]
#     sum += int(batch.ano_num[i])

#     annotations = batch.ano_x[ano_start:ano_end]
#     origin = batch.x[o_start:o_end]

#     true_label = []
#     for i, a in enumerate(annotations):
#         for o in origin:
#             if torch.equal(a, o):
#                 true_label.append(batch.y[i])
#                 break
#     true_label = torch.tensor(true_label)
#     print(true_label.shape)

# print(batch)
# for i in loader:
#     print(i)
# for i in batch:
#     print(i)
# batch = next(iter(loader))
# print(batch.ptr)


# # # loader.get(0)
