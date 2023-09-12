#!/usr/bin/env python
# coding: utf-8

# In[1]:
import open3d as o3d
import pickle
import numpy as np
import time
import copy
import os
import sys
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'

# In[2]:

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


# In[3]:

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# In[4]:


def prepare_dataset(voxel_size, source_point_cloud, target_point_cloud):
    source = source_point_cloud
    target = target_point_cloud
    trans_init = np.eye(4)
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


# In[5]:


def create_o3d_point_cloud(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    return point_cloud


# In[6]:


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
#     print(":: Apply fast global registration with distance threshold %.3f" \
#             % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

# ## Getting correspondences between sequential point clouds

# In[17]:


def create_pointcloud_for_learning_dataset_from_sapien():
    folder = f"{THIS_DIR}/PointClouds/"
    files = sorted(list(os.listdir(folder)))
    min_num_corr = np.inf
    num_particles = 40
    num_controls = 50
    samples_per_control = 5
    data = np.zeros(
        (len(files) // (samples_per_control * num_controls),
         samples_per_control - 1,
         num_controls, 2, 5, num_particles)
    )

    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[2]):
            for k in range(data.shape[1]):
                index1 = 40 * 5 * i + 5 * j + k
                index2 = index1 + 1
                source_pcl = np.load(folder + files[index1])
                target_pcl = np.load(folder + files[index2])
                tauL, tauR = source_pcl[0][-2:]
                source_point_cloud = create_o3d_point_cloud(source_pcl[:, :3])  # last two are controls
                target_point_cloud = create_o3d_point_cloud(target_pcl[:, :3])  # last two are controls
                voxel_size = 0.15
                source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
                    voxel_size, source_point_cloud, target_point_cloud)
                result_fast = execute_fast_global_registration(source_down, target_down,
                                                           source_fpfh, target_fpfh,
                                                           voxel_size)
                correspondence = np.asarray(result_fast.correspondence_set)[:num_particles]
                min_num_corr = min(min_num_corr, len(correspondence))
                source_pts = np.asarray(source_down.points)[correspondence[:, 0]].T
                target_pts = np.asarray(target_down.points)[correspondence[:, 1]].T
                controls = np.full((2, len(correspondence)), fill_value=np.array([tauL, tauR]).reshape((2, 1)))
                source_pts = np.append(source_pts, controls, axis = 0)
                target_pts = np.append(target_pts, controls, axis = 0)
                correspondence_points = np.stack([source_pts, target_pts], axis = 0)
                try:
                    data[i, k, j] = correspondence_points
                except ValueError as e:
                    print("Insufficient data")
                    data[i, k, j] = np.zeros((2, 5, num_particles))

    np.save(f"{THIS_DIR}/PointCloudData.npy",data)


def create_pointcloud_for_learning_from_real_jackal():
    base_dir = f"{THIS_DIR}/PointCloudRealJackalData"
    traj_folders = [folder for folder in os.listdir(base_dir) if "traj_" in folder]
    N = len(traj_folders)
    S = 30  # num of samples
    P = 400  # num of particles
    voxel_size = 0.15
    min_num_corr = np.inf

    final_array = np.zeros((N, 1, S, 2, 5, P))  # Last dimension is 5 for x,y,z,u1,u2

    for i, folder in tqdm(enumerate(sorted(traj_folders))):
        folder_path = os.path.join(base_dir, folder)
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])[:S]

        for j in tqdm(range(len(files) - 1)):
            source_pcl = np.load(os.path.join(folder_path, files[j]))
            target_pcl = np.load(os.path.join(folder_path, files[j + 1]))
            tauL, tauR = source_pcl[0][-2:]
            source_point_cloud = create_o3d_point_cloud(source_pcl[:, :3])  # last two are controls
            target_point_cloud = create_o3d_point_cloud(target_pcl[:, :3])  # last two are controls
            source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,
                                                                                                 source_point_cloud,
                                                                                                 target_point_cloud)
            result_fast = execute_fast_global_registration(source_down, target_down,
                                                           source_fpfh, target_fpfh,
                                                           voxel_size)
            correspondence = np.asarray(result_fast.correspondence_set)[:P]
            min_num_corr = min(min_num_corr, len(correspondence))

            source_pts = np.asarray(source_down.points)[correspondence[:, 0]]
            target_pts = np.asarray(target_down.points)[correspondence[:, 1]]

            controls = np.full((2, len(correspondence)), fill_value=np.array([tauL, tauR]).reshape((2, 1))).T

            source_pts = np.append(source_pts, controls, axis=1)
            target_pts = np.append(target_pts, controls, axis=1)
            correspondence_points = np.stack([source_pts, target_pts], axis=1)

            if (result_fast.fitness < 0.2):
                print(f'({i}, {j}): correspondence_points = {len(correspondence)}, score = {result_fast.fitness}')

            try:
                final_array[i, :, j, :, :, :] = correspondence_points.reshape((2, 5, 400))
            except ValueError as e:
                print("Insufficient data")
                remaining = P - len(correspondence_points)
                additional_correspondences = np.repeat(correspondence_points, remaining, axis=0)
                correspondence_points = np.concatenate([correspondence_points, additional_correspondences[:remaining]])
                final_array[i, :, j, :, :, :] = correspondence_points.reshape((2, 5, 400))

            # # DRAW ###########################3
            # if (i == 0):
            #     source_points = o3d.geometry.PointCloud()
            #     source_points.points = o3d.utility.Vector3dVector(final_array[i, 0, :, 0, 0:3])
            #     target_points = o3d.geometry.PointCloud()
            #     target_points.points = o3d.utility.Vector3dVector(final_array[i, 0, :, 1, 0:3])
            #     draw_registration_result(source_points, target_points, result_fast.transformation)
            #     print(f'T = {result_fast.transformation}')
            # ###############################

    save_path = os.path.join(THIS_DIR, "real_point_cloud_dataset.npy")
    np.save(save_path, final_array)
    print(f'SAVED point_cloud_dataset: {final_array.shape}')


if __name__ == '__main__':
    # create_pointcloud_for_learning_dataset_from_sapien()
    create_pointcloud_for_learning_from_real_jackal()
