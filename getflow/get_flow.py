#get_flow
import os
import sys


sys.path.append('./RAFT/')
sys.path.append('./RAFT/core')
sys.path.append('./data/')

from dis_index import FlowEstimator, cosine_project_ratio

import cv2
import argparse
import os.path as osp
import numpy as np
from tqdm import tqdm
from itertools import combinations

checkpoint = './RAFT/models/raft-things.pth'
sdi_name = 'flow'
time_name = "bezier"
flow_estimator = FlowEstimator(checkpoint=checkpoint)

def save_flow(sample_path, name, context, i0, i1):
    save_path = osp.join(sample_path, '{}_{}_{}.npy'.format(name, i0, i1))
    with open(save_path, 'wb') as f:
        np.save(f, context.astype(np.half))

def cosine_project_ratio_with_three_imgs(img1_path, img2_path, img3_path):
    img1_to_img2, _ = flow_estimator.estimate_flow(img1_path, img2_path)
    img1_to_img3, _ = flow_estimator.estimate_flow(img1_path, img3_path)
    img1_to_img2 = img1_to_img2[0].permute(1, 2, 0).cpu().numpy()
    img1_to_img3 = img1_to_img3[0].permute(1, 2, 0).cpu().numpy()
    B_pred = 2*img1_to_img2 - 0.5*img1_to_img3
    return img1_to_img2, img1_to_img3, B_pred


def create_dis_index_for_dataset(root, sample_paths, avg=False, downsample_ratio=2., sample_length=7):
    base_path = osp.join(root, "sequences")
    for sample_path in tqdm(sample_paths, total=len(sample_paths)):
        sample_path = sample_path.strip()
        img_paths = [osp.join(osp.join(base_path, sample_path), 'im{}.png'.format(i + 1)) for i in range(sample_length)]
        combs = list(combinations(list(range(sample_length)), r=3))

        for comb in combs:
            img1_path = img_paths[comb[0]]
            img2_path = img_paths[comb[1]]
            img3_path = img_paths[comb[2]]
            img_resized_dis_index = cosine_project_ratio_with_three_imgs(img1_path, img2_path, img3_path)
            img_resized_dis_index_inv = cosine_project_ratio_with_three_imgs(img3_path, img2_path, img1_path)
            sample_path = osp.join(base_path, sample_path)
            #save_flow(sample_path, sdi_name, img_resized_dis_index[0], 1, 2)
            #save_flow(sample_path, sdi_name, img_resized_dis_index[1], 1, 3)
            save_flow(sample_path, time_name, img_resized_dis_index[2], 1, 3)
            
            #save_flow(sample_path, sdi_name, img_resized_dis_index_inv[0], 3, 2)
            #save_flow(sample_path, sdi_name, img_resized_dis_index_inv[1], 3, 1)
            save_flow(sample_path, time_name, img_resized_dis_index_inv[2], 3, 1)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_list_path', type=str)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--sample_length', type=int, default=3)
    args = parser.parse_args()
    print('sample_list_path:', args.sample_list_path)
    with open(osp.join(args.root_path, args.sample_list_path)) as f:
        sample_paths = f.readlines()
    create_dis_index_for_dataset(root = args.root_path, sample_paths=sample_paths,
                                 sample_length=args.sample_length)
