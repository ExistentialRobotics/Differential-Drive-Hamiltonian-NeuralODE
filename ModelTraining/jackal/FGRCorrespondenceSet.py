#!/usr/bin/env python
# coding: utf-8
import open3d as o3d
import numpy as np
import copy
import os
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'


def draw_registration_result(source, target, transformation):
    """
    draw the point cloud registration result using given transformation matrix
    """
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


def preprocess_point_cloud(pcd, voxel_size):
    """
    process the point cloud to estimate point cloud normals, and downsample the pointclouds for faster processing
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source_point_cloud, target_point_cloud):
    """
    return downsample source, target pointclouds and their feature vectors
    """
    source = source_point_cloud
    target = target_point_cloud
    trans_init = np.eye(4)
    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def create_o3d_point_cloud(points):
    """
    create open3d point cloud object from a numpy array of points
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.reshape((-1, 3)))
    return point_cloud


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    """
    input: downsampled source, target pointcloud, source pointcloud features, target pointcloud features
    output: fast global registration result on source and target pointclouds
    """
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def create_pointcloud_for_learning_dataset_from_sapien():
    """
    Take the pointcloud created using sapien during different trajectory collection
    Use fast global registration during successive point clouds within trajectory and create a correspondence pointcloud dataset
    """
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
                index1 = num_controls * samples_per_control * i + samples_per_control * j + k
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
                source_pts = np.append(source_pts, controls, axis=0)
                target_pts = np.append(target_pts, controls, axis=0)
                correspondence_points = np.stack([source_pts, target_pts], axis=0)
                try:
                    data[i, k, j] = correspondence_points
                except (ValueError, RuntimeError) as e:
                    print("Insufficient data")
                    data[i, k, j] = np.zeros((2, 5, num_particles))

    np.save(f"{THIS_DIR}/PointCloudData.npy", data)


if __name__ == '__main__':
    create_pointcloud_for_learning_dataset_from_sapien()
