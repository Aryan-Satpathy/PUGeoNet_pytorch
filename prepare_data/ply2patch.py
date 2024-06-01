import numpy as np
import os
import open3d as o3d
from pc_util import extract_knn_patch, normalize_point_cloud, farthest_point_sample, group_points,group_points_with_normal

import multiprocessing as mp

raw_data_path='/home/aryan/baselines/PUGeo/mesh'
save_root_path='/home/aryan/baselines/PUGeo/pytorch_data'

basic_num=5000
num_point_per_patch=256
num_patch=int(basic_num/num_point_per_patch * 3)

def call_worker2(i, mode, up_ratio, input_data_name, centroid, furthest_distance, centroid_points, furthest_distance_patches, centroid_patches):
    label_data=np.loadtxt(os.path.join(os.path.join(raw_data_path, '%s_%d' % (mode, int(basic_num*up_ratio)),input_data_name)))
    xyz=(label_data[:,0:3]-centroid)/furthest_distance
    label_data[:,0:3]=xyz
    label_patches=group_points_with_normal(label_data,centroid_points,num_point_per_patch*up_ratio)
    label_patches_xyz=(label_patches[:,:,0:3]-centroid_patches)/furthest_distance_patches
    label_patches[:,:,0:3]=label_patches_xyz
    np.save(os.path.join(save_root_path,'%d'%up_ratio,'%d.npy'%i),label_patches)
    print(input_data_name,up_ratio)

def call_worker(i, up_ratio_list, mode, input_data_name) : 
    # call([sys.executable, os.path.join(os.path.dirname(__file__), 'beesalgoalter.py'), '--log', '--useNFE'] + args)
    input_data = np.loadtxt(os.path.join(os.path.join(raw_data_path, '%s_%d' % (mode, int(basic_num)),input_data_name)))
    xyz,centroid,furthest_distance=normalize_point_cloud(input_data[:,0:3])
    input_data[:,0:3]=xyz
    centroid_points=farthest_point_sample(input_data,num_patch)
    input_patches=group_points_with_normal(input_data,centroid_points,num_point_per_patch)
    normalized_input_patches_xyz,centroid_patches,furthest_distance_patches=normalize_point_cloud(input_patches[:,:,0:3])
    input_patches[:,:,0:3]=normalized_input_patches_xyz
    np.save(os.path.join(save_root_path,'basic','%d.npy'%i),input_patches)
    
    # pool = mp.Pool(5)

    results = []
    for up_ratio in up_ratio_list:
        results.append(call_worker2(i, mode, up_ratio, input_data_name, centroid, furthest_distance, centroid_points, furthest_distance_patches, centroid_patches))

    # results = [result.get() for result in results]
    
    return

def build_dataset(up_ratio_list,mode):
    for up_ratio in up_ratio_list:
        try:
            os.mkdir(os.path.join(save_root_path,'%d'%up_ratio))

        except:
            pass
    try:
        os.mkdir(os.path.join(save_root_path, 'basic'))
    except:
        pass

    input_data_path_list =os.listdir(os.path.join(raw_data_path, '%s_%d' % (mode, int(basic_num))))
    #label_data_path_list =os.listdir( os.path.join(raw_data_path,'%s_%d'%(mode,int(basic_num*up_ratio))))
    i=0

    pool = mp.Pool(60)

    results = []
    for input_data_name in input_data_path_list:
        results.append(pool.apply_async(call_worker, (i, up_ratio_list, mode, input_data_name)))# lambda x : bar.update))
    
    pool.close()
    pool.join()

    results = [result.get() for result in results]
        

if __name__=='__main__':
    up_ratio_list = [4, 8, 12, 16]
    build_dataset(up_ratio_list,'train')
