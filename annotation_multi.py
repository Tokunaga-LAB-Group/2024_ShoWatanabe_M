import argparse
import csv
import glob
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.spatial import KDTree
from tqdm import tqdm

from partnet import CATEGORIES


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'True', 'true', 'yes', '1'}:
        return True
    elif value.lower() in {'False', 'false', 'no', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_csv", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--target_names", nargs="*", type=str, required=True)
    parser.add_argument("--annotation_per", type=float, default=0.1)
    parser.add_argument("--train_split", type=float, default=0.5)
    parser.add_argument("--use_sampling", type=str2bool, default=False)
    parser.add_argument("--annotation_type", type=str, required=True) # simple, const, random, rev
    parser.add_argument("--annotation_point", type=str, required=True) # boundary, interior, random

    return parser.parse_args()


args = get_arguments()


ANNOTATION_PERCENT = args.annotation_per


# 重複なし
def rand_ints_nodup(max, num):
    ns = []
    while len(ns) < num:
        n = random.randint(0, max)
        if not n in ns:
            ns.append(n)
    return ns

import random


def generate_indices(max_value: int, num: int):
    """
    0..(max_value-1) の整数を合計 num 個返す。
    まず各整数を (num // max_value) 回ずつ入れ、
    残りの (num % max_value) 個をランダムに選んで1回ずつ追加する。

    例:
        max_value = 10, num = 25
         -> 各数字(0..9)を2回ずつ=20個,
            さらに5つをランダムに1回ずつ=5個,
            計25個のリストを返す。
    """
    # 1. num // max_value 回ずつ全数字を入れる
    if max_value == 0:
        return []
    repeat = num // max_value
    mod = num % max_value

    ns = []
    for i in range(max_value):
        ns.extend([i] * repeat)  # iを repeat 回追加

    # 2. 余りの個数だけ distinct なインデックスを選んで1回ずつ追加
    chosen = random.sample(range(max_value), mod)
    for c in chosen:
        ns.append(c)

    # 3. リスト全体をシャッフル（必要に応じて）
    random.shuffle(ns)

    return ns

def generate_indices_by_fps(points, num):
    max_value = len(points)

    # 1. num // max_value 回ずつ全数字を入れる
    if max_value == 0:
        return []
    repeat = num // max_value
    mod = num % max_value

    ns = []
    for i in range(max_value):
        ns.extend([i] * repeat)  # iを repeat 回追加

    # 2. 余りの個数だけ distinct なインデックスを選んで1回ずつ追加
    _, _, _, indexes = farthest_point_sample(points, [key] * max_value, mod)
    for c in indexes:
        ns.append(c)


    # 3. リスト全体をシャッフル（必要に応じて）
    random.shuffle(ns)

    return ns

# 重複なし
def get_unique_random_pairs(point_dict, num_pairs):
    # 各キーとその要素のペアをリストとして取得
    all_pairs = [(key, item) for key, values in point_dict.items() for item in values]

    # 取得したいペア数が、総ペア数を超えていないか確認
    num_pairs = min(num_pairs, len(all_pairs))

    # ペアをランダムにシャッフルし、指定した数だけ取得
    random.shuffle(all_pairs)
    selected_pairs = all_pairs[:num_pairs]

    return selected_pairs

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
        return [], [], False, []
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
    return point, label, True, centroids.astype(np.int32)

def extract_boundary_points(points, labels, k=200):
    """
    完全アノテーションされた点群からクラス間境界付近の点を抽出する関数。

    引数:
        points: (N,2)のnumpy配列。点群座標。
        labels: (N,)のnumpy配列。各点のクラスID（整数または文字列）。
        k: 各点周囲の近傍点数。大きすぎると計算量増加、小さすぎると局所的になりすぎる。

    戻り値:
        boundary_degrees: 境界度
    """
    tree = KDTree(points)
    N = points.shape[0]
    boundary_degrees = np.zeros(N, dtype=float)

    # 各点について近傍点を取得
    _, indices = tree.query(points, k=k+1)  # 最初に自点が返るのでk+1

    # 自点を除外して解析
    neighbor_indices = indices[:, 1:]

    # 各点に対して異クラス近傍があるか判定
    for i in range(N):
        base_label = labels[i]
        neigh_labels = labels[neighbor_indices[i]]
        boundary_degree = 0
        for j in range(len(neigh_labels)):
            if base_label != neigh_labels[j]:
                boundary_degree += 1 - (j/k)
        boundary_degrees[i] = boundary_degree / k

    return boundary_degrees

def annotated_by_simple(point_dict):
    num = 0
    annotations = {}
    total_points = sum(len(points) for points in point_dict.values())
    for key, value in point_dict.items():
        anno_num = int(len(value) * ANNOTATION_PERCENT)
        if num == len(point_dict) - 1:
            total_anno_points = sum(annotations.values())
            # if args.use_sampling:
            anno_num = int(total_points * ANNOTATION_PERCENT) - total_anno_points
        annotations[int(key)] =  anno_num
        num += 1

    return annotations

def annotated_by_const2(point_dict):
    # 全体の合計点数
    total_points = sum(len(points) for points in point_dict.values())
    # 全体で選択したい点数
    total_anno_num = int(total_points * ANNOTATION_PERCENT)

    # 各クラスに最初に割り当てる点数
    initial_allocations = {}
    for key, points in point_dict.items():
        initial_allocations[key] = int(total_anno_num / len(point_dict))

    return initial_allocations

def annotated_by_const(point_dict):
    # 全体の合計点数
    total_points = sum(len(points) for points in point_dict.values())
    # 全体で選択したい点数
    total_anno_num = int(total_points * ANNOTATION_PERCENT)

    # 各クラスに最初に割り当てる点数
    initial_allocations = {}
    for key, points in point_dict.items():
        initial_allocations[key] = int(total_anno_num / len(point_dict))

    # 各クラスの選択数を調整 (不足分を他のクラスに再分配)
    remaining_points = total_anno_num
    allocations = {}

    # 各クラスに選べる最大数を割り当て
    for key, points in point_dict.items():
        alloc = min(initial_allocations[key], len(points))
        allocations[key] = alloc
        remaining_points -= alloc

    # 残りの点数を他のクラスで再分配
    while remaining_points > 0:
        for key, points in point_dict.items():
            if remaining_points == 0:
                break
            if allocations[key] < len(points):
                allocations[key] += 1
                remaining_points -= 1
    return allocations

def annotated_by_random(point_dict, cls_num):
    total_points = sum(len(points) for points in point_dict.values())
    total_anno_num = int(total_points * ANNOTATION_PERCENT)
    # Start with zero items in each class
    annotations = [0] * cls_num

    # Randomly assign items to classes
    while sum(annotations) < total_anno_num:
        selected_cls = np.random.randint(0, cls_num)
        if annotations[selected_cls] < len(point_dict[selected_cls]):
            annotations[selected_cls] += 1

    return annotations

import numpy as np


def annotated_by_random2(point_dict, cls_num):
    """
    クラス毎のポイント数を出現確率としてランダムに選択し、アノテーションを割り当てる。

    Parameters:
    - point_dict (dict): クラスIDをキー、ポイントのリストを値とする辞書。
    - cls_num (int): クラス数。
    - annotation_percent (float): アノテーション数の割合（例: 0.01 で 1%）。

    Returns:
    - annotations (list): 各クラスに割り当てられたアノテーションの数。
    """
    # 全ポイント数を計算
    total_points = sum(len(points) for points in point_dict.values())

    # 総アノテーション数を計算
    total_anno_num = int(total_points * ANNOTATION_PERCENT)

    # クラスごとのポイント数を基に確率分布を計算
    class_probabilities = np.array([len(point_dict.get(cls, [])) for cls in range(cls_num)])
    class_probabilities = class_probabilities / class_probabilities.sum()

    # 各クラスに割り当てるアノテーション数
    annotations = [0] * cls_num

    # ランダムにアノテーションを割り当て
    while sum(annotations) < total_anno_num:
        selected_cls = np.random.choice(cls_num, p=class_probabilities)
        if annotations[selected_cls] < len(point_dict[selected_cls]):
            annotations[selected_cls] += 1

    return annotations


def annotated_by_rev(point_dict, cls_num, total_percentage=1, epsilon=1):
    class_counts = [0] * cls_num # 大きめに
    for key, value in point_dict.items():
        class_counts[int(key)] = len(value)

    total_points = sum(class_counts)
    total_annotations = int(total_percentage / 100 * total_points)

    nokori = class_counts
    diff = total_annotations
    annotations = [0] * len(class_counts)
    valid_indices = [i for i in range(len(class_counts))]
    while diff > 6:
        # 反比例する重みを計算
        weights = [1 / (count + epsilon) for count in nokori]
        # 割り当て計算
        weighted_annotations = [w / sum(weights) * diff for w in weights]
        # 割り当て中のアノテーションのみに適用
        for index, i in enumerate(valid_indices):
            annotations[i] += np.floor(weighted_annotations[index]).astype(int)
        invalid_indices = [i for i, count in enumerate(class_counts) if annotations[i] >= count]
        valid_indices = [i for i, count in enumerate(class_counts) if annotations[i] < count]

        for cls in invalid_indices:
            annotations[cls] = class_counts[cls]

        # 残りのアノテーション数
        diff = total_annotations - sum(annotations)

        nokori = [elem - annotations[i] for i, elem in enumerate(class_counts) if i not in invalid_indices]

    # 残りはランダムに割り当て
    while diff > 0:
        for idx in valid_indices:
            if diff <= 0:
                break
            annotations[idx] += 1
            diff -= 1

    return annotations

def annotated_by_rev2(point_dict, cls_num, total_percentage=1, epsilon=1):
    class_counts = [0] * cls_num
    for key, value in point_dict.items():
        class_counts[int(key)] = len(value)

    total_points = sum(class_counts)
    total_annotations = int(total_percentage / 100 * total_points)

    # 反比例する重みを計算
    weights = [1 / (count + epsilon) if count > 0 else 0 for count in class_counts]

    # 各クラスに割り当てる数を計算
    weighted_annotations = [w / sum(weights) * total_annotations for w in weights]

    # 各クラスのアノテーション数を丸める
    annotations = [int(round(w)) for w in weighted_annotations]

    # 合計が一致するように調整
    diff = total_annotations - sum(annotations)
    if diff > 0:
        # アノテーションを追加
        for i in sorted(range(len(annotations)), key=lambda x: annotations[x]):
            if class_counts[i] > 0:  # class_counts が 0 のクラスを除外
                annotations[i] += 1
                diff -= 1
                if diff == 0:
                    break
    elif diff < 0:
        # アノテーションを削減（アノテーション数が0より大きいクラスのみ対象）
        for i in sorted(range(len(annotations)), key=lambda x: -annotations[x]):
            if class_counts[i] > 0 and annotations[i] > 0:  # class_counts が 0 のクラスを除外
                annotations[i] -= 1
                diff += 1
                if diff == 0:
                    break

    return annotations

with open(args.annotation_csv, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "x", "y", "z", "label"])
    targets = args.target_names
    cls_num = 2
    sampling_num = 2800

    for k in range(len(targets)):
        if args.target_names[k] == "Airplane":
            cls_num = 4
            sampling_num = 2400
        target = CATEGORIES[args.target_names[k]]["id"]
        label_dir = "{}/{}/expert_verified/points_label".format(
            args.dataset_dir, target
        )
        point_dir = "{}/{}/points".format(args.dataset_dir, target)
        label_paths = natsorted(glob.glob(label_dir + "/*"))
        boundary_index = int(len(label_paths) * args.train_split)
        for p in tqdm(label_paths[:boundary_index]):
            label_file_name = p.split("/")[-1].split(".")[0]

            labels = []
            points = []
            with open(p, "r") as file:
                label_lines = file.readlines()
                for line in label_lines:
                    labels.append(int(line) - 1 + k * 2)

            with open(f"{point_dir}/{label_file_name}.pts", "r") as file:
                point_lines = file.readlines()
                for i in range(len(point_lines)):
                    points.append(list(map(float, point_lines[i].split(" "))))

            if args.use_sampling:
                points, labels, ok, _ = farthest_point_sample(points, labels, 2400)
                if not ok:
                    continue

            point_dict = defaultdict(list)
            for i in range(len(labels)):
                point_dict[labels[i]].append(points[i])


            anno_nums = []
            annotations = {}
            if args.annotation_type == "simple":
                annotations = annotated_by_simple(point_dict)
            elif args.annotation_type == "const":
                annotations = annotated_by_const(point_dict)
            elif args.annotation_type == "const2":
                annotations = annotated_by_const2(point_dict)
            elif args.annotation_type == "random":
                annotations = annotated_by_random(point_dict, cls_num)
            elif args.annotation_type == "random2":
                annotations = annotated_by_random2(point_dict, cls_num)
            elif args.annotation_type == "rev":
                annotations = annotated_by_rev(point_dict, cls_num, total_percentage=ANNOTATION_PERCENT*100, epsilon=50)
            else:
                annotations = annotated_by_rev2(point_dict, cls_num, total_percentage=ANNOTATION_PERCENT*100, epsilon=50)

            # sum_anno = 0
            # for i in range(len(annotations)):
            #     # print(f"Class {i}: {annotations[i]}")
            #     sum_anno += annotations[i]
            # if label_file_name == "2af529843a47df7aba0d990ae229b477":
            #     print(f"Total: {sum_anno},  {annotations}")

            # print(f"Total: {sum_anno},  {annotations}")
            # print("=========================")

            if args.annotation_point == "boundary" or args.annotation_point == "interior":
                boundary_degrees = extract_boundary_points(np.array(points), np.array(labels))

            for key, value in point_dict.items():
                anno_num = annotations[int(key)]
                indexes = []
                if args.annotation_point == "random":
                    indexes = generate_indices(len(value), anno_num)

                    for i in indexes:
                        writer.writerow(
                            [
                                label_file_name,
                                point_dict[key][i][0],
                                point_dict[key][i][1],
                                point_dict[key][i][2],
                                key,
                            ]
                        )
                elif args.annotation_point == "uniformed":
                    indexes = generate_indices_by_fps(value, anno_num)
                    for i in indexes:
                        writer.writerow(
                            [
                                label_file_name,
                                point_dict[key][i][0],
                                point_dict[key][i][1],
                                point_dict[key][i][2],
                                key,
                            ]
                        )
                elif args.annotation_point == "boundary":
                    idx = np.where(np.array(labels) == key)[0]
                    cls_boundary_degrees = boundary_degrees[idx]
                    sorted_idx = np.argsort(-cls_boundary_degrees)  # 降順ソート
                    if len(sorted_idx) < anno_num:
                        # sorted_idx の繰り返しと追加
                        repeat_count = anno_num // len(sorted_idx)
                        remainder_count = anno_num % len(sorted_idx)

                        # 繰り返し分を追加
                        repeated_indexes = np.tile(sorted_idx, repeat_count)

                        # 残り分を昇順で取得
                        remainder_indexes = np.sort(sorted_idx)[:remainder_count]

                        # 繰り返し分と残り分を結合
                        final_indexes = np.concatenate((repeated_indexes, remainder_indexes))
                    else:
                        # sorted_idx[:anno_num] をそのまま使用
                        final_indexes = sorted_idx[:anno_num]

                    indexes = idx[final_indexes]
                    for i in indexes:
                        writer.writerow(
                            [
                                label_file_name,
                                points[i][0],
                                points[i][1],
                                points[i][2],
                                key,
                            ]
                        )
                elif args.annotation_point == "interior":
                    idx = np.where(labels == key)[0]
                    cls_boundary_degrees = boundary_degrees[idx]
                    sorted_idx = np.argsort(cls_boundary_degrees)
                    if len(sorted_idx) < anno_num:
                        # sorted_idx の繰り返しと追加
                        repeat_count = anno_num // len(sorted_idx)
                        remainder_count = anno_num % len(sorted_idx)

                        # 繰り返し分を追加
                        repeated_indexes = np.tile(sorted_idx, repeat_count)

                        # 残り分を昇順で取得
                        remainder_indexes = np.sort(sorted_idx)[:remainder_count]

                        # 繰り返し分と残り分を結合
                        final_indexes = np.concatenate((repeated_indexes, remainder_indexes))
                    else:
                        # sorted_idx[:anno_num] をそのまま使用
                        final_indexes = sorted_idx[:anno_num]

                    indexes = idx[final_indexes]
                    for i in indexes:
                        writer.writerow(
                            [
                                label_file_name,
                                points[i][0],
                                points[i][1],
                                points[i][2],
                                key,
                            ]
                        )
