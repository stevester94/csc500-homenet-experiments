import numpy as np
import pickle
import os
from definitions import *

def load_from_disk(dataset_index):
    root_dir = './'
    if dataset_index == WIFI_PREAM:
        dataset_name = 'grid_2019_12_25.pkl'
    elif dataset_index == WIFI2_PREAM:
        dataset_name = 'grid_2020_02_03.pkl'
    elif dataset_index == WIFI3_PREAM:
        dataset_name = 'grid_2020_02_04.pkl'
    elif dataset_index == WIFI4_PREAM:
        dataset_name = 'grid_2020_02_05.pkl'
    elif dataset_index == WIFI5_PREAM:
        dataset_name = 'grid_2020_02_06.pkl'
        
    else:
        raise ValueError('Wrong dataset index')
    dataset_path = root_dir + dataset_name
    with open(dataset_path,'rb') as f:
        dataset = pickle.load(f)
    return dataset





def fetch_mdatasets():
    dataset1 = load_from_disk(WIFI_PREAM)
    dataset1['Index'] = WIFI_PREAM
    dataset2 = load_from_disk(WIFI2_PREAM)
    dataset2['Index'] = WIFI2_PREAM
    dataset3 = load_from_disk(WIFI3_PREAM)
    dataset3['Index'] = WIFI3_PREAM
    dataset4 = load_from_disk(WIFI4_PREAM)
    dataset4['Index'] = WIFI4_PREAM
    dataset5 = load_from_disk(WIFI5_PREAM)
    dataset5['Index'] = WIFI5_PREAM
    dataset =  merge_datasets([dataset1,dataset2,dataset3,dataset4])
    test_dataset=dataset5
    return (dataset,test_dataset)


# Merge multiple datasets
def merge_datasets(datasets):
    # Merge node lists
    full_list = []
    for d in datasets:
        full_list = full_list + d['node_list']
    full_list = sorted(list(set(full_list)))
    
    # Initialize destination dataset 
    full_dataset = {}
    full_dataset['node_list'] = full_list
    full_dataset['data'] = [[] for i in range(len(full_list))]
    full_dataset['data'] 
    # Place data in lists for each node
    for d in datasets:
        for src_i,src_n in enumerate(d['node_list']):
            dest_i = full_dataset['node_list'].index(src_n)
            full_dataset['data'][dest_i].append(d['data'][src_i])
    
    # Merge lists for each node and shuffle
    for i,d in enumerate(full_dataset['data']):
        if len(d)==1:
            dd = d[0]
        else:
            dd = np.concatenate(d)
        full_dataset['data'][i] = dd
        np.random.shuffle(full_dataset['data'][i])
    return full_dataset

# normalize data
def norm(sig_u):
    if len(sig_u.shape)==3:
        pwr = np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr[:,None,None]
    if len(sig_u.shape)==2:
        pwr = np.sqrt(np.mean(sig_u**2,axis = -1))
        sig_u = sig_u/pwr[:,None]
    # print(sig_u.shape)
    return sig_u