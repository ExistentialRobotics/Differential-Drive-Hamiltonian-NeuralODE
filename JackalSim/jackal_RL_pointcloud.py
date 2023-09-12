import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import itertools as it
from PIL import Image, ImageColor
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2quat, quat2euler, euler2mat
import time
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import gym
from gym.utils import seeding
import copy
from collections import defaultdict
from tqdm import tqdm
from energy_based_new_lyapunov import EnergyBasedController
import open3d
vis = open3d.visualization.Visualizer()
vis.create_window(visible = False)

def create_jackal_ackerman(scene: sapien.Scene,
                           body_size=(0.5, 0.5, 0.25),
                           radius=0.1,
                           joint_friction=0,
                           joint_damping=0,
                           density=100
                           ) -> sapien.Articulation:
    body_half_size = np.array(body_size) / 2
    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    # car body
    car_body: sapien.LinkBuilder = builder.create_link_builder()
    car_body.set_name('jackal')
    car_body.add_box_collision(half_size=body_half_size, density=density)
    car_body.add_box_visual(half_size=body_half_size, color=[0.8, 0.6, 0.4])

    # front shaft
    front_shaft = builder.create_link_builder(car_body)
    front_shaft.set_name('front_shaft')
    front_shaft.set_joint_name('front_shaft_joint')
    front_shaft.add_box_collision(pose=sapien.Pose(q=euler2quat(np.deg2rad(90), 0, 0)),
                                  half_size=(0.25 * radius, 0.25 * radius, body_half_size[2] + 0.45), density=density)
    front_shaft.add_box_visual(pose=sapien.Pose(q=euler2quat(np.deg2rad(90), 0, 0)),
                               half_size=(0.25 * radius, 0.25 * radius, body_half_size[2] + 0.45), color=(1, 0, 0))
    front_shaft.set_joint_properties(
        'fixed',
        limits=[],
        pose_in_parent=sapien.Pose(p=[-body_half_size[0] + radius, 0, -body_half_size[2] + radius / 2],
                                   q=euler2quat(0, np.deg2rad(90), 0)),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, np.deg2rad(90), 0)),
        friction=joint_friction,
        damping=joint_damping
    )

    # rear right wheel
    rr_wheel = builder.create_link_builder(front_shaft)
    rr_wheel.set_name('rear_right_wheel')
    rr_wheel.add_sphere_collision(radius=radius, density=density)
    rr_wheel.add_sphere_visual(radius=radius, color=[0, 1, 0])
    rr_wheel.set_joint_name('rr_wheel_joint')
    rr_wheel.set_joint_properties(
        'revolute',
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(p=[0, -body_half_size[2] - 0.45, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping
    )

    # rear left wheel
    rl_wheel = builder.create_link_builder(front_shaft)
    rl_wheel.set_name('rear_left_wheel')
    rl_wheel.add_sphere_collision(radius=radius, density=density)
    rl_wheel.add_sphere_visual(radius=radius, color=[0, 1, 0])
    rl_wheel.set_joint_name('rl_wheel_joint')
    rl_wheel.set_joint_properties(
        'revolute',
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(p=[0, body_half_size[2] + 0.45, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping
    )

    # back  shaft
    back_shaft = builder.create_link_builder(car_body)
    back_shaft.set_name('back_shaft')
    back_shaft.set_joint_name('back_shaft_joint')
    back_shaft.add_box_collision(pose=sapien.Pose(q=euler2quat(np.deg2rad(90), 0, 0)),
                                 half_size=(0.25 * radius, 0.25 * radius, body_half_size[2] + 0.45), density=density)
    back_shaft.add_box_visual(pose=sapien.Pose(q=euler2quat(np.deg2rad(90), 0, 0)),
                              half_size=(0.25 * radius, 0.25 * radius, body_half_size[2] + 0.45), color=(1, 0, 0))
    back_shaft.set_joint_properties(
        'fixed',
        limits=[],
        pose_in_parent=sapien.Pose(p=[body_half_size[0] - radius, 0, -body_half_size[2] + radius / 2],
                                   q=euler2quat(0, np.deg2rad(90), 0)),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, np.deg2rad(90), 0)),
        friction=joint_friction,
        damping=joint_damping
    )

    # front right wheel
    fr_wheel = builder.create_link_builder(back_shaft)
    fr_wheel.set_name('front_right_wheel')
    fr_wheel.add_sphere_collision(radius=radius, density=density)
    fr_wheel.add_sphere_visual(radius=radius, color=[0, 1, 0])
    fr_wheel.set_joint_name('fr_wheel_joint')
    fr_wheel.set_joint_properties(
        'revolute',
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(p=[0, -body_half_size[2] - 0.45, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping
    )

    # front left wheel
    fl_wheel = builder.create_link_builder(back_shaft)
    fl_wheel.set_name('front_left_wheel')
    fl_wheel.add_sphere_collision(radius=radius, density=density)
    fl_wheel.add_sphere_visual(radius=radius, color=[0, 1, 0])
    fl_wheel.set_joint_name('fl_wheel_joint')
    fl_wheel.set_joint_properties(
        'revolute',
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(p=[0, body_half_size[2] + 0.45, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping
    )
    jackal = builder.build()
    jackal.set_name('jackal_robot')
    return jackal


def get_joints_dict(articulation: sapien.Articulation):
    joints = articulation.get_joints()
    joint_names = [joint.name for joint in joints]
    joint_types = [joint.type for joint in joints]
    joint_mode = [joint.drive_mode for joint in joints]
    print(joint_types)
    print(joint_mode)
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--static-friction', default=0.05, type=float, help='static friction')
    parser.add_argument('--dynamic-friction', default=0.05, type=float, help='dynamic friction')
    parser.add_argument('--restitution', default=0.1, type=float, help='restitution (elasticity of collision)')
    parser.add_argument('--joint-friction', default=0.25, type=float, help='joint friction')
    parser.add_argument('--joint-damping', default=0.25, type=float,
                        help='joint damping (resistance proportional to joint velocity)')
    parser.add_argument('--samples_per_control', default=5, type=int, help='Number of sample steps per control')
    parser.add_argument('--num_controls', default=50, type=int, help='Number of control samples')
    parser.add_argument('--num_trajectories', default=22, type=int, help='Number of different trajectories')

    args = parser.parse_args()
    return args


class SapienEnv(gym.Env):
    def __init__(self, control_freq, timestep):
        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep
        scene_config = sapien.SceneConfig()
        scene_config.default_static_friction = args.static_friction
        scene_config.default_dynamic_friction = args.dynamic_friction
        scene_config.default_restitution = args.restitution
        self._engine = sapien.Engine()
        self._renderer = sapien.VulkanRenderer()
        self._engine.set_renderer(self._renderer)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        self._scene.set_timestep(1 / 20)

        # self._build_world()
        self.viewer = None
        self.seed()

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            pass  # release viewer

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self._setup_viewer()
            self._scene.update_render()
            self.viewer.render()
        else:
            raise NotImplementedError('Unsupported render mode {}.'.format(mode))

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_actor(self, name) -> sapien.ArticulationBase:
        all_actors = self._scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f'Not a unique name for actor: {name}')
        elif len(actor) == 0:
            raise RuntimeError(f'Actor not found: {name}')
        return actor[0]

    def get_articulation(self, name) -> sapien.ArticulationBase:
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f'Not a unique name for articulation: {name}')
        elif len(articulation) == 0:
            raise RuntimeError(f'Articulation not found: {name}')
        return articulation[0]

    @property
    def dt(self):
        return self.timestep * self.control_freq


class JackalEnv(SapienEnv):
    def __init__(self):
        super().__init__(control_freq=5, timestep=0.01)
        self._build_world()
        self._setup_viewer()
        self.point_cloud_dict = defaultdict(list)
        self.controller = EnergyBasedController(maxTorque=5)

    def _build_world(self):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._scene.add_ground(0.0)
        self.jackal = create_jackal_ackerman(self._scene)
        self.jackal.set_pose(sapien.Pose([15, 10, 0.22], [1, 0, 0, 0]))
        city = loader.load('assets/demo.urdf')
        city.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        near, far = 0.1, 50
        width, height = 640, 480
        self.camera_actor_1 = self._scene.create_actor_builder().build_kinematic()
        self.camera_1 = self._scene.add_mounted_camera(
            name="camera",
            actor=self.camera_actor_1,
            pose=sapien.Pose(),
            width=width, height=height, fovy=np.deg2rad(80), near=near, far=far
        )
        print(f'Camera intrinsic matrix : {self.camera_1.get_intrinsic_matrix()}')

        self.joints = get_joints_dict(self.jackal)
        print(self.joints.keys())
        self.joints['front_shaft_joint'].set_drive_property(stiffness=5, damping=0)
        self.joints['fl_wheel_joint'].set_drive_property(stiffness=0, damping=0.5)
        self.joints['fr_wheel_joint'].set_drive_property(stiffness=0, damping=0.5)
        self.joints['rl_wheel_joint'].set_drive_property(stiffness=0, damping=0.5)
        self.joints['rr_wheel_joint'].set_drive_property(stiffness=0, damping=0.5)

    def _setup_viewer(self):
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=20, y=0, z=1)
        self.viewer.set_camera_rpy(r=0, p=-0.3, y=3.14)
        self.viewer.focus_entity(self.jackal)


    def step(self, target, i):
        jackal_pose_cam1 = np.array([[1, 0, 0, 0.48], [0, 1, 0, -0.25], [0, 0, 1, 0.36], [0, 0, 0, 1]])
        jackal_pose_cam2 = np.array([[1, 0, 0, 0.48], [0, 1, 0, 0.25], [0, 0, 1, 0.36], [0, 0, 0, 1]])
        samples_per_control = args.samples_per_control
        num_controls = args.num_controls
        dim_SE3 = 20
        dataset_SE3 = np.zeros((samples_per_control, num_controls, dim_SE3))
        start = time.time()
        pose = self.jackal.get_pose()
        mat = sapien.Pose.to_transformation_matrix(pose)
        currentPos = np.array([mat[:3, -1]]).reshape(-1, )
        currentRot = mat[:3, :3]
        currentVel = np.zeros(3, )
        currentOmega = np.zeros(3, )

        desiredPos = np.array([target[0], target[1], 0.175]).reshape(-1, )
        desiredRot = euler2mat(0, 0, target[-1])
        desiredVel = np.zeros(3, )
        desiredOmega = np.zeros(3, )
        self.viewer.render()
        for j in tqdm(range(num_controls)):
            passive_force = self.jackal.compute_passive_force(True, True, False)
            currentState = np.concatenate(
                [
                    currentPos,
                    currentRot.flatten(),
                    currentVel,
                    currentOmega
                ],
                axis=0
            )
            targetState = np.concatenate(
                [
                    desiredPos,
                    desiredRot.flatten(),
                    desiredVel,
                    desiredOmega,
                ],
                axis=0
            )
            torque, done = self.controller.get_control(currentState=currentState,
                                                       targetState=targetState
                                                       )
            tauL, tauR = torque
            torques = np.array([tauL, tauR, tauL, tauR])[::-1].reshape(-1, )  # right torque, left torque
            # torques = np.array([-0.5, 0.5, -0.5, 0.5])
            for s in range(samples_per_control):
                jackal_pose = self.jackal.get_pose()
                links = self.jackal.get_links()
                v, w = links[0].get_velocity(), links[0].get_angular_velocity()
                mat = sapien.Pose.to_transformation_matrix(jackal_pose)
                R = mat[:3, :3]
                v = R.T @ np.array(v).reshape(-1)
                w = np.array(w).reshape(-1)
                x, y, z = mat[:3, -1]
                currentPos = np.array([x, y, z]).reshape(-1, )
                currentRot = mat[:3, :3]
                currentVel = v
                currentOmega = w
                r1, r2, r3 = mat[0, :3], mat[1, :3], mat[2, :3]
                dataset_SE3[s, j, :] = [x, y, z,
                                    *r1, *r2, *r3,
                                    v[0], v[1], 0,
                                    0, 0, w[2],
                                    tauL, tauR]
                world_pose_cam1 = mat @ jackal_pose_cam1
                world_pose_cam1 = sapien.Pose.from_transformation_matrix(world_pose_cam1)
                self.camera_actor_1.set_pose(world_pose_cam1)
                self.camera_1.take_picture()

                # Actor level segmented image
                seg_labels = self.camera_1.get_uint32_texture('Segmentation')
                label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level

                # Getting point cloud of scene using any one camera
                position = self.camera_1.get_float_texture('Position')
                points_opengl = position[..., :3][label1_image > 1]

                # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
                model_matrix = self.camera_1.get_model_matrix()

                points_world = (points_opengl @ model_matrix[:3, :3].T).astype(np.float32)[::2]
                points_world = np.append(
                    points_world,
                    np.full(
                        (points_world.shape[0], 2),
                        fill_value=np.array([tauL, tauR]).reshape((1, 2))
                    ), axis=1
                )
                # np.save(
                #     "../ModelTraining/jackal/data/PointClouds/{:06}.npy".format(
                #         num_controls * samples_per_control * i + samples_per_control * j + s
                #     ),
                #     points_world
                # )

                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(np.array(points_world[:, :3]).reshape([-1, 3]))
                #
                open3d.visualization.draw_geometries([point_cloud])
                self.jackal.set_qf(torques)
                self._scene.step()
                self._scene.update_render()
                self.viewer.render()

        tspan = np.arange(samples_per_control) * self.dt
        self.reset()
        return dataset_SE3, tspan

    def return_physical_props(self):
        links = self.jackal.get_links()
        mass = sum([m.get_mass() for m in links])
        inertial = sum([m.get_inertia() for m in links])
        print(mass)
        print(inertial)
        return mass, inertial

    def reset(self):
        p_mean = np.array([15, -3, 0.175])
        theta_mean = 0
        p = np.zeros(3)
        p[:2] = np.random.normal(0, 1, 1) + p_mean[:2]
        p[2] = p_mean[2]
        q = euler2quat(0, 0, theta_mean)
        init_qpos = np.array([0, 0, 0, 0])
        init_qvel = np.array([0, 0, 0, 0])
        self.jackal.set_qpos(init_qpos)
        self.jackal.set_qvel(init_qvel)
        self.jackal.set_pose(sapien.Pose(p=p_mean, q=q))


def get_dataset(test_split=0.5, save_dir=None, generate=True, **kwargs):
    data = {}
    fix_root_link = kwargs['fix_root_link']
    balance_passive_force = kwargs['balance_passive_force']
    assert save_dir is not None
    path_SE3 = '{}/jackal-SE3-pointclouds-dataset.pkl'.format(save_dir)
    if not generate:
        data_SE3 = from_pickle(path_SE3)
        print("Successfully loaded data from {}".format(path_SE3))
    else:
        print("Had a problem loading data from {}. Rebuilding dataset...".format(path_SE3))
        # waypoints_x = np.array([13, 14, 16, 17])
        # waypoints_y = np.array([-1, 1])
        # waypoints_theta = np.array([-np.pi/3, -np.pi/6, np.pi/6, np.pi/3])

        waypoints_x = np.array([15])
        waypoints_y = np.array([-3])
        waypoints_theta = np.array([0])

        setPoints = list(it.product(waypoints_x, waypoints_y, waypoints_theta))
        num_trajectories = len(setPoints)

        Jackal = JackalEnv()
        samples_per_control = args.samples_per_control
        num_controls = args.num_controls
        dim_SE3 = 20
        dataset_SE3 = np.zeros((num_trajectories, samples_per_control, num_controls, dim_SE3), dtype=np.float32)
        point_clouds_dict = {}
        Jackal.reset()
        for i in tqdm(range(num_trajectories)):
            data_SE3, tspan = Jackal.step(target=setPoints[i], i=i)
            dataset_SE3[i, :, :, :] = data_SE3
            Jackal.reset()
            time.sleep(1.5)

        split_data_SE3 = {}
        samples = dataset_SE3.shape[2]
        split_ix = int(samples * test_split)
        split_data_SE3['x'], split_data_SE3['test_x'] = dataset_SE3[:, :, :split_ix, :], dataset_SE3[:, :, split_ix:, :]
        data_SE3 = split_data_SE3
        data_SE3['t'] = tspan
        to_pickle(data_SE3, path_SE3)


if __name__ == '__main__':
    args = parse_args()
    get_dataset(0.8,
                '../ModelTraining/jackal/data',
                generate=True,
                fix_root_link=True,
                balance_passive_force=True
                )
