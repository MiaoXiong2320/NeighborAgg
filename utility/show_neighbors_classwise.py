import numpy as np

import os
import os.path as osp

from sklearn.neighbors import KDTree
import pdb

import argparse
import os
import os.path as osp
import json
import shutil
import random, time
import faiss


def main(
    dist_measure,
    src_folder,
    K):

    res = faiss.StandardGpuResources() 
    # K = 20
    FETCH_ALL_CLASS = True # use to indicate fetching neighbors from the same class only OR from every classes 

    #src_folder = "../classifier/mnist_pretrained/baseline/feats/"
    # src_folder = "../classifier/code/simple_cls/output/mnist3digits_smallerconvnet/feats/"
    os.makedirs(src_folder, exist_ok=True)
    train_feats = np.load(osp.join(src_folder, "train_feats.npy"))
    train_ys = np.load(osp.join(src_folder, "train_ys.npy"))
    val_feats = np.load(osp.join(src_folder, "val_feats.npy"))
    val_ys = np.load(osp.join(src_folder, "val_ys.npy"))
    test_feats = np.load(osp.join(src_folder, "test_feats.npy"))
    test_ys = np.load(osp.join(src_folder, "test_ys.npy"))

    # val_feats = np.load(osp.join(src_folder, "ood_val_feats.npy"))
    # val_ys = np.load(osp.join(src_folder, "ood_val_ys.npy"))

    kdtree_dict = {} # from y to KDTree(train_feats)
    kdtree_index = {}

    # traverse every samples' class label
    start = time.time()
    for train_y in train_ys:
        if not train_y in kdtree_dict:
            inds = np.where(train_ys == train_y)[0] # find all samples belong to this label:train_y
            train_feats_y = train_feats[inds, :]    # fetch corresponding features

            len_feature = train_feats.shape[1]
            if dist_measure == "L2":
                neigh_searcher = faiss.IndexFlatL2(len_feature) # build a KDTree
            elif dist_measure == 'Dot':
                neigh_searcher = faiss.IndexFlatIP(len_feature)
            
            neigh_searcher = faiss.index_cpu_to_gpu(res, 3, neigh_searcher)
            neigh_searcher.add(train_feats_y)
            kdtree_dict[train_y] = neigh_searcher
            kdtree_index[train_y] = inds  # record this KDTree samples' index
    end = time.time()
    print("Building KDTree:{:.2f} mins".format((end-start)/60) )

    num_classes = len(kdtree_dict)

    ####### FOR TRAINING SET ########
    neigh_indices_list = [] 
    neigh_dists_list = []

    # traverse every validation samples' class label
    start = time.time()
    for (i, train_y) in enumerate(train_ys):
        #inds = np.where(train_ys == train_y)[0]
        #train_feats_trainy = train_feats[inds, :]
        if FETCH_ALL_CLASS:
            neigh_dists_all = np.empty((1,num_classes*K), dtype=float)
            neigh_indices_all = np.empty((1,num_classes*K), dtype=float)
            for cls in range(num_classes):
                kdtree = kdtree_dict[cls] 
                neigh_dists, neigh_indices = kdtree.search(train_feats[i][np.newaxis, ...], k=K+1) # ---> [1, K]
                neigh_dists = neigh_dists[:,1:K+1]
                neigh_indices = neigh_indices[:,1:K+1]
                neigh_indices = kdtree_index[cls][neigh_indices] # transform to true index in the whole training dataset
                assert (train_ys[neigh_indices] == cls).all()
                neigh_dists_all[0, cls*K:(cls+1)*K] = neigh_dists
                neigh_indices_all[0, cls*K:(cls+1)*K] = neigh_indices
        else:
            kdtree = kdtree_dict[train_y] 
            neigh_dists, neigh_indices = kdtree.search(train_feats[i][np.newaxis, ...], k=K+1) # ---> [1, K]
            neigh_dists = neigh_dists[:,1:K+1]
            neigh_indices = neigh_indices[:,1:K+1]
            neigh_indices = kdtree_index[train_y][neigh_indices] # transform to true index in the whole training dataset instead of that in the partial train_y's sub training dataset
            assert (train_ys[neigh_indices] == train_y).all()
            neigh_indices_all = neigh_indices
            neigh_dists_all = neigh_dists
        
        neigh_indices_list.append(neigh_indices_all)
        neigh_dists_list.append(neigh_dists_all)

    neigh_indices_table = np.concatenate(neigh_indices_list)
    neigh_dists_table = np.concatenate(neigh_dists_list)

    end = time.time()
    print("Finding Training Neighbors: {:.2f} mins".format((end-start)/60) )

    np.save(osp.join(src_folder, "train2train_idx_allclass.npy"), neigh_indices_table)
    np.save(osp.join(src_folder, "train2train_dist_allclass.npy"), neigh_dists_table)

    ####### FOR VALIDATION SET ########
    neigh_indices_list = [] 
    neigh_dists_list = []

    # traverse every validation samples' class label
    start = time.time()
    for (i, val_y) in enumerate(val_ys):
        #inds = np.where(train_ys == val_y)[0]
        #train_feats_valy = train_feats[inds, :]
        if FETCH_ALL_CLASS:
            neigh_dists_all = np.empty((1,num_classes*K), dtype=float)
            neigh_indices_all = np.empty((1,num_classes*K), dtype=float)
            for cls in range(num_classes):
                kdtree = kdtree_dict[cls] 
                neigh_dists, neigh_indices = kdtree.search(val_feats[i][np.newaxis, ...], k=K) # ---> [1, K]
                neigh_indices = kdtree_index[cls][neigh_indices] # transform to true index in the whole training dataset
                assert (train_ys[neigh_indices] == cls).all()
                neigh_dists_all[0, cls*K:(cls+1)*K] = neigh_dists
                neigh_indices_all[0, cls*K:(cls+1)*K] = neigh_indices
        else:
            kdtree = kdtree_dict[val_y] 
            neigh_dists, neigh_indices = kdtree.search(val_feats[i][np.newaxis, ...], k=K) # ---> [1, K]
            neigh_indices = kdtree_index[val_y][neigh_indices] # transform to true index in the whole training dataset instead of that in the partial val_y's sub training dataset
            assert (train_ys[neigh_indices] == val_y).all()
            neigh_indices_all = neigh_indices
            neigh_dists_all = neigh_dists
        
        neigh_indices_list.append(neigh_indices_all)
        neigh_dists_list.append(neigh_dists_all)

    neigh_indices_table = np.concatenate(neigh_indices_list)
    neigh_dists_table = np.concatenate(neigh_dists_list)

    end = time.time()
    print("Finding Validation Neighbors: {:.2f} mins".format((end-start)/60) )

    np.save(osp.join(src_folder, "val2train_idx_allclass.npy"), neigh_indices_table)
    np.save(osp.join(src_folder, "val2train_dist_allclass.npy"), neigh_dists_table)

    ########### FOR TEST DATASET ##########
    neigh_indices_list = [] 
    neigh_dists_list = []

    # traverse every validation samples' class label
    start = time.time()
    for (i, test_y) in enumerate(test_ys):
        #inds = np.where(train_ys == test_y)[0]
        #train_feats_valy = train_feats[inds, :]
        if FETCH_ALL_CLASS:
            neigh_dists_all = np.empty((1,num_classes*K), dtype=float)
            neigh_indices_all = np.empty((1,num_classes*K), dtype=float)
            for cls in range(num_classes):
                kdtree = kdtree_dict[cls] 
                neigh_dists, neigh_indices = kdtree.search(test_feats[i][np.newaxis, ...], k=K) # ---> [1, K]
                neigh_indices = kdtree_index[cls][neigh_indices] # transform to true index in the whole training dataset
                assert (train_ys[neigh_indices] == cls).all()
                neigh_dists_all[0, cls*K:(cls+1)*K] = neigh_dists
                neigh_indices_all[0, cls*K:(cls+1)*K] = neigh_indices
        else:
            kdtree = kdtree_dict[test_y] 
            neigh_dists, neigh_indices = kdtree.search(test_feats[i][np.newaxis, ...], k=K) # ---> [1, K]
            neigh_indices = kdtree_index[test_y][neigh_indices] # transform to true index in the whole training dataset instead of that in the partial test_y's sub training dataset
            assert (train_ys[neigh_indices] == test_y).all()
            neigh_indices_all = neigh_indices
            neigh_dists_all = neigh_dists
        
        neigh_indices_list.append(neigh_indices_all)
        neigh_dists_list.append(neigh_dists_all)

    neigh_indices_table = np.concatenate(neigh_indices_list)
    neigh_dists_table = np.concatenate(neigh_dists_list)

    end = time.time()
    print("Finding Test Neighbors: {:.2f} mins".format((end-start)/60) )

    np.save(osp.join(src_folder, "test2train_idx_allclass.npy"), neigh_indices_table)
    np.save(osp.join(src_folder, "test2train_dist_allclass.npy"), neigh_dists_table)    

    #np.save(osp.join(src_folder, "smallconvnet_val2train_classwise.npy"), neigh_indices_table)
    # np.save(osp.join(src_folder, "val2train_allclass.npy"), neigh_indices_table)
    # np.save(osp.join(src_folder, "val2train_dist_allclass.npy"), neigh_dists_table)

    # np.save(osp.join(src_folder, "val2train_ood_allclass.npy"), neigh_indices_table)
    # np.save(osp.join(src_folder, "val2train_dist_ood_allclass.npy"), neigh_dists_table)

    print("DONE.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_measure", type=str, default="L2", help="similarity measure")
    parser.add_argument("--src_folder", type=str, help="../classifier/mnist_pretrained/baseline/feats/")
    parser.add_argument("--K", type=int, default=20)

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)

