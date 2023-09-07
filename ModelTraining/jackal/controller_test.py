import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import os
import sys
from transforms3d.quaternions import mat2quat
from transforms3d.euler import euler2quat, quat2euler, mat2euler, euler2mat
from energy_based_new_lyapunov import EnergyBasedController
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PARENT_DIR)


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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--M1', type=int, default=10, help='Number of layers in mass network')
    parser.add_argument('--M2', type=int, default=5, help='Number of layers in inertial network')
    parser.add_argument('--g', type=int, default=10, help='Number of layers in control matrix network')
    parser.add_argument('--D', type=int, default=5, help='Number of layers in friction matrix network')
    parser.add_argument('--M1_known', nargs='?', type=int, default=0, help="Train assuming M1 known")
    parser.add_argument('--useD', nargs='?', type=int, default=0, help="Train with friction")
    parser.add_argument('--static-friction', default=0.05, type=float, help='static friction')
    parser.add_argument('--dynamic-friction', default=0.05, type=float, help='dynamic friction')
    parser.add_argument('--restitution', default=0.1, type=float, help='restitution (elasticity of collision)')
    parser.add_argument('--joint-friction', default=0.25, type=float, help='joint friction')
    parser.add_argument('--joint-damping', default=0.25, type=float,
                        help='joint damping (resistance proportional to joint velocity)')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Checkpoint model')
    args = parser.parse_args()
    return args


def get_joints_dict(articulation: sapien.Articulation):
    joints = articulation.get_joints()
    joint_names = [joint.name for joint in joints]
    joint_types = [joint.type for joint in joints]
    joint_mode = [joint.drive_mode for joint in joints]
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}


def ref_traj(mode, t, xc, yc, r, f):
    result = None
    if mode == 'circle':
        x = xc + r * np.cos(f * t)
        y = yc + r * np.sin(f * t)
        vx = -f * r * np.sin(f * t)
        vy = f * r * np.cos(f * t)
        th = f * t
        omega = f
        ax = -(f ** 2) * r * np.cos(f * t)
        ay = -(f ** 2) * r * np.sin(f * t)
        result = [x, y, vx, vy, th, omega, ax, ay]
    return result


def main(args, fix_root_link=True, xdes: float = 5, ydes: float = 5, thdes: float = np.pi/4):
    args = args

    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene_config.default_static_friction = args.static_friction
    scene_config.default_dynamic_friction = args.dynamic_friction
    scene_config.default_restitution = args.restitution
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 20.0)
    # scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    scene.add_ground(altitude=0)
    jackal = create_jackal_ackerman(scene)
    jackal.set_pose(sapien.Pose([0, 0, 0.175], [1, 0, 0, 0]))

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=20, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=3.14)
    viewer.focus_entity(jackal)
    steps = 0

    controller = EnergyBasedController(M1_known=False, M2_known=False, maxTorque=5, checkpoint=args.checkpoint)
    # controller = EnergyBasedController(M1_known=False, M2_known=False, maxTorque=5)
    links = jackal.get_links()
    applied = 0
    joints = get_joints_dict(jackal)
    joints['front_shaft_joint'].set_drive_property(stiffness=5, damping=0)
    joints['fl_wheel_joint'].set_drive_property(stiffness=0, damping=0.92)
    joints['fr_wheel_joint'].set_drive_property(stiffness=0, damping=0.92)
    joints['rl_wheel_joint'].set_drive_property(stiffness=0, damping=0.92)
    joints['rr_wheel_joint'].set_drive_property(stiffness=0, damping=0.92)
    passive_force = jackal.compute_passive_force(True, True, False)
    steps = 0
    used_PE = None
    folder = None
    start = time.time()
    control = np.zeros((4, 1))
    done = False
    trajectory = np.empty((0, 3))
    done = False
    qd = dict()
    for steps in tqdm(range(10_000)):
        if done:
            break
        # passive_force = jackal.compute_passive_force(True, True, False)
        pose = jackal.get_pose()
        qd["pos"] = np.array(pose.p).reshape(-1, )
        # qd["pos"] = np.array([0, 0, pose.p[2]]).reshape(-1, )
        rot = sapien.Pose.to_transformation_matrix(pose)[:3, :3]
        qd["rot"] = rot
        v, w = links[0].get_velocity(), links[0].get_angular_velocity()
        v = rot.T @ v
        w = rot.T @ w
        qd["vel"] = np.array(v)
        qd["omega"] = np.array(w)
        trajectory = np.append(trajectory, np.array([pose.p[0], pose.p[1], mat2euler(rot)[-1]]).reshape((-1, 3))
                               , axis=0)
        currentState = np.concatenate(
            [
                qd["pos"],
                qd["rot"].flatten(),
                qd["vel"],
                qd["omega"]
            ],
            axis=0
        )

        # qd["pos_des"] = np.array([args.xdes, args.ydes, 0.175])
        qd["pos_des"] = np.array([xdes, ydes, pose.p[2]]).reshape(-1, )
        qd["vel_des"] = np.array([0, 0, 0])
        qd["omega_des"] = np.array([0, 0, 0])
        qd["rot_des"] = euler2mat(0, 0, thdes)
        qd["pos_des_dot"] = np.array([0, 0, 0])
        qd["pos_des_ddot"] = np.array([0, 0, 0])
        controls = np.zeros(4, )
        targetState = np.concatenate(
            [
                qd["pos_des"],
                qd["rot_des"].flatten(),
                qd["vel_des"],
                qd["omega_des"]
            ],
            axis=0
        )
        torques, done = controller.get_control_new(
            currentState=currentState,
            targetState=targetState
        )
        controls = np.array([torques[0], torques[1], torques[0], torques[1]])[::-1]  # right torque, left torque
        print(controls)
        # print(f"desired orientation of robot : {mat2euler(qd['rot_des'])[-1]}")
        # print(f"current yaw of robot : {mat2euler(qd['rot'])[-1]}")
        # for _ in range(5):
        jackal.set_qf(controls)
        scene.step()
        scene.update_render()
        viewer.render()
    # viewer.close()
    return trajectory


if __name__ == '__main__':
    args = parse_args()
    # x = np.linspace(-3, 3, 5)
    # y = np.linspace(-3, 3, 5)
    # th = np.linspace(np.pi/2, np.pi/2, 5)
    x = [2]
    y = [-2]
    th = [-0*np.pi/3]
    waypoints = list(product(x, y, th))
    for it in tqdm(range(len(waypoints))):
        xdes, ydes, thdes = waypoints[it]
        trajectory = main(args, True, *waypoints[it])
        fig, axs = plt.subplots(2, 2)
        N = len(trajectory)
        axs[0, 0].plot(range(N), trajectory[:, 0], 'r', label='x')
        axs[0, 0].plot(range(N), [xdes]*N, 'b--', label='desired x')
        axs[0, 0].set_xlabel("Iterations")
        axs[0, 0].set_ylabel("X Coordinate")
        axs[0, 0].legend(loc="upper right")

        axs[0, 1].plot(range(N), trajectory[:, 1], 'r', label='y')
        axs[0, 1].plot(range(N), [ydes]*N, 'b--', label='desired y')
        axs[0, 1].set_xlabel("Iterations")
        axs[0, 1].set_ylabel("Y Coordinate")
        axs[0, 1].legend(loc="upper right")

        axs[1, 0].plot(range(N), trajectory[:, 2], 'r', label=r'$\theta$')
        axs[1, 0].plot(range(N), [thdes]*N, 'b--', label='desired' + r' $\theta$')
        axs[1, 0].set_xlabel("Iterations")
        axs[1, 0].set_ylabel("Yaw")
        axs[1, 0].legend(loc="upper right")

        axs[1, 1].scatter(trajectory[:, 0], trajectory[:, 1], s=0.08, label='trajectory')
        axs[1, 1].set_xlabel("X coordinate")
        axs[1, 1].set_ylabel("Y Coordinate")
        axs[1, 1].legend(loc="upper right")
        plt.savefig("{}/png/x={:.3f} y={:.3f} theta={:.3f}.png".format(THIS_DIR, xdes, ydes, thdes))
        plt.show(block=True)

        # ToDo --> Fix multiple trajectory tests with controller in sapien. currently crashes after 2 sims

