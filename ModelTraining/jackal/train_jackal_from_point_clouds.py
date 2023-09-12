# This code was borrowed and modified from the following repository:
# https://github.com/thaipduong/SE3HamDL
#
# The original code was written by Thai Duong.
# 
# 
import torch
import argparse
import pickle
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchdiffeq import odeint, odeint_adjoint
from utils import traj_pose_L2_geodesic_loss, pose_L2_geodesic_loss, point_cloud_l2_loss_for_successive_point_clouds
from scipy.spatial.transform import Rotation
from scipy.linalg import logm
from se3hamneuralode import SE3HamNODE, from_pickle
from data_collection import get_dataset, arrange_data
from tqdm import tqdm

# ************* DIRECTORY ************* #  
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'

# ************* PARAMETERS ************* #
mass = 6.77
J_list = [[1.05, 0.0, 0.0], [0.0, 1.05, 0.0], [0.0, 0.0, 2.05]]
J = torch.tensor(J_list)


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--learn_rate', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    parser.add_argument('--total_steps', default=500, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=10, type=int, help='number of gradient steps between prints')
    parser.add_argument('--name', default='jackal', type=str, help='only one option right now')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='where to save the trained model')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_points', type=int, default=5,
                        help='number of evaluation points by the ODE solver, including the initial point')
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.add_argument('--M1_known', default=0, type=int, help='Assume mass known')
    parser.add_argument('--M2_known', default=0, type=int, help='Assume inertia known')
    parser.add_argument('--control_freq_hz', default=48.0, type=float, help='frequency of control in HZ')
    parser.add_argument('--samples_per_control', default=5, type=int, help='number of samples per control action')
    return parser.parse_args()


def train(args, trainPointCloudSeq, testPointCloudSeq, t_eval):
    """
    trainDataset: [num_trajs, samples_per_control, num_controls, 20]  # 20 = p (3) + R(12) + v(3) + w(3) + u(2)
    trainPointCloudSeq: [num_trajs, samples_per_control, num_controls, 2, 5, P]  # last two for controls
    testPointCloudSeq: [num_trajs, samples_per_control, num_controls, 2, 5, P]  # last two for controls
    """
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    print(f'DEVICE: {device}')

    num_actions = trainPointCloudSeq.shape[2]
    print(f'num_actions = {num_actions}')
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)

    # For data from real jackal, dt = 0.1, number of timesteps = 1
    # t_eval = np.array([0, 0.1])
    # t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)
    print(f't_eval = {t_eval}')

    pc_train = trainPointCloudSeq
    pc_train_cat = pc_train
    pc_train_cat = torch.tensor(pc_train_cat, requires_grad=True, dtype=torch.float32).to(device)
    pc_test = testPointCloudSeq
    pc_test_cat = pc_test
    pc_test_cat = torch.tensor(pc_test_cat, requires_grad=True, dtype=torch.float32).to(device)

    model = SE3HamNODE(
        device=device,
        udim=2,
        pretrain=True,
        M1_known=bool(args.M1_known),
        M2_known=bool(args.M2_known)
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=0.0)
    stats = {'train_loss': [], 'test_loss': []}

    for step in tqdm(range(args.total_steps + 1)):
        if step % 100 == 0 and step > 0:
            for gr in optim.param_groups:
                gr['lr'] *= 0.5
        batch = pc_train_cat.shape[0]
        # Train dataset 
        target_hat = None
        for u in range(pc_train_cat.shape[2]):
            if u == 0:
                init_state = torch.tensor([0, 0, 0,
                                           1, 0, 0, 0, 1, 0, 0, 0, 1,
                                           0, 0, 0,
                                           0, 0, 0],
                                          requires_grad=True, dtype=torch.float32).to(device)
                # Repeat for batch size
                init_state = init_state.repeat(batch, 1)
                x0 = torch.cat((init_state, pc_train_cat[:, 0, u, 0, -2:, 0]), dim=1)
                y_pred = odeint_adjoint(
                    model,
                    x0,
                    t_eval,
                    method=args.solver
                )  # (samples_per_control, num_trajectories, num_state_variables)
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = y_pred
            else:
                y0 = torch.cat((target_hat[:, -1, u - 1, 0:18], pc_train_cat[:, 0, u, 0, -2:, 0]), dim=1)
                y_pred = odeint_adjoint(model, y0, t_eval, method=args.solver)
                y_pred = y_pred[:, :, :]
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = torch.cat((target_hat, y_pred), dim=2)
        target_hat = target_hat.permute(1, 0, 2, 3)
        target_hat = target_hat.flatten(1, 2)
        target = pc_train_cat.permute(1, 0, 2, 3, 4, 5)
        target = target.flatten(1, 2)  # (samples_per_control - 1) X (num_trajectories X num_controls) X 2 X 5 X num_particles

        train_loss = point_cloud_l2_loss_for_successive_point_clouds(
            observedPointClouds=target,
            predictedStates=target_hat,
            split=[model.xdim, model.Rdim, model.twistdim, model.udim]
        )

        gnet_l1_loss = 0
        lambda_gnet_l1_loss = 1e-3
        rand_sample_indx = np.random.randint(0, target_hat.shape[0])
        rand_control_indx = np.random.randint(1, target_hat.shape[1])
        q = target_hat[rand_sample_indx, rand_control_indx, :12]
        gnet_l1_loss += torch.abs(model.g_net(q)).mean()
        train_loss += (lambda_gnet_l1_loss * gnet_l1_loss)

        # Test dataset
        target_hat = None
        batch = pc_test_cat.shape[0]
        for u in range(pc_test_cat.shape[2]):
            if u == 0:
                init_state = torch.tensor([0, 0, 0,
                                           1, 0, 0, 0, 1, 0, 0, 0, 1,
                                           0, 0, 0,
                                           0, 0, 0],
                                          requires_grad=True, dtype=torch.float32).to(device)
                init_state = init_state.repeat(batch, 1)
                x0 = torch.cat((init_state, pc_test_cat[:, 0, u, 0, -2:, 0]), dim=1)
                y_pred = odeint_adjoint(model, x0, t_eval, method=args.solver)  # (5, 18, 24)
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = y_pred
            else:
                y0 = torch.cat((target_hat[:, -1, u - 1, 0:18], pc_test_cat[:, 0, u, 0, -2:, 0]), dim=1)
                y_pred = odeint_adjoint(model, y0, t_eval, method=args.solver)
                y_pred = y_pred[:, :, :]
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = torch.cat((target_hat, y_pred), dim=2)
        target_hat = target_hat.permute(1, 0, 2, 3)
        target_hat = target_hat.flatten(1, 2)
        target = pc_test_cat.permute(1, 0, 2, 3, 4, 5)
        target = target.flatten(1, 2)

        test_loss = point_cloud_l2_loss_for_successive_point_clouds(
            observedPointClouds=target,
            predictedStates=target_hat,
            split=[model.xdim, model.Rdim, model.twistdim, model.udim]
        )

        # Save loss 
        stats['train_loss'].append(train_loss.item())
        stats['test_loss'].append(test_loss.item())

        if step > 0:
            train_loss.backward()
            optim.step()
            optim.zero_grad()

        if step % args.print_every == 0:
            print(f'step ({step}): train_loss_mini = {train_loss}')
            print(f'step ({step}): test_loss_mini = {test_loss}')
            print(f'---------------------------------------------')
            os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None
            label = '-se3ham_pointclouds'
            path = '{}/PointCloudTrainedModels/{}{}-{}-{}p-{}.tar'.format(args.save_dir, args.name, label, args.solver,
                                                                          args.num_points, step)
            torch.save(model.state_dict(), path)

    # Evaluate
    train_data_hat = None
    batch = pc_train_cat.shape[0]
    num_samples = pc_train_cat.shape[2]
    num_particles = pc_train_cat.shape[-1]
    for i in range(num_samples - 1):
        if i == 0:
            init_state = torch.tensor([0, 0, 0,
                                       1, 0, 0, 0, 1, 0, 0, 0, 1,
                                       0, 0, 0,
                                       0, 0, 0],
                                      requires_grad=True, dtype=torch.float32).to(device)
            # Repeat for batch size
            init_state = init_state.repeat(batch, 1)
            # Concatenate with control input
            x0 = torch.cat((init_state, pc_train_cat[:, 0, i, 0, -2:, 0]), dim=1)
            x_hat = odeint(model, x0, t_eval, method=args.solver)
            train_data_hat = torch.unsqueeze(x_hat[-1, :, :], dim=0)

        else:
            x0 = torch.cat((train_data_hat[i - 1, :, 0:18], pc_train_cat[:, 0, i, 0, -2:, 0]), dim=1)  # (batch, 20)
            x_hat = odeint(model, x0, t_eval, method=args.solver)
            train_data_hat = torch.cat((train_data_hat, torch.unsqueeze(x_hat[-1, :, :], dim=0)), dim=0)

    # Test data evaluation
    test_data_hat = None
    batch = pc_test_cat.shape[0]
    num_samples = pc_test_cat.shape[2]
    num_particles = pc_test_cat.shape[-1]
    for i in range(num_samples - 1):
        if i == 0:
            init_state = torch.tensor([0, 0, 0,
                                       1, 0, 0, 0, 1, 0, 0, 0, 1,
                                       0, 0, 0,
                                       0, 0, 0],
                                      requires_grad=True, dtype=torch.float32).to(device)
            # Repeat for batch size
            init_state = init_state.repeat(batch, 1)
            # Concatenate with control input
            x0 = torch.cat((init_state, pc_test_cat[:, 0, i, 0, -2:, 0]), dim=1)
            x_hat = odeint(model, x0, t_eval, method=args.solver)
            test_data_hat = torch.unsqueeze(x_hat[-1, :, :], dim=0)

        else:
            x0 = torch.cat((test_data_hat[i - 1, :, 0:18], pc_test_cat[:, 0, i, 0, -2:, 0]), dim=1)  # (batch, 20)
            x_hat = odeint(model, x0, t_eval, method=args.solver)
            test_data_hat = torch.cat((test_data_hat, torch.unsqueeze(x_hat[-1, :, :], dim=0)), dim=0)

    stats['train_x'] = pc_train_cat.detach().cpu().numpy()
    stats['test_x'] = pc_test_cat.detach().cpu().numpy()
    stats['train_x_hat'] = train_data_hat.detach().cpu().numpy()
    stats['test_x_hat'] = test_data_hat.detach().cpu().numpy()
    stats['t_eval'] = t_eval.detach().cpu().numpy()

    return model, stats


def saveModelStats(model, stats):
    os.makedirs(THIS_DIR) if not os.path.exists(THIS_DIR) else None
    label = '-se3ham_pointclouds'
    path = '{}/PointCloudTrainedModels/{}{}-{}-{}p.tar'.format(THIS_DIR, 'jackal', label, 'rk4', '5')
    torch.save(model.state_dict(), path)
    path = '{}/PointCloudTrainedModels/{}{}-{}-{}p-stats.pkl'.format(THIS_DIR, 'jackal', label, 'rk4', '5')
    print("Saved file: ", path)
    with open(path, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    args = get_args()
    point_cloud_data = np.load(f"{THIS_DIR}/PointCloudData.npy")
    num_controls = point_cloud_data.shape[2]
    split_idx = int(num_controls * 0.8)
    train_point_cloud_sequence = point_cloud_data[:, :, :split_idx, :, :, :]
    test_point_cloud_sequence = point_cloud_data[:, :, split_idx:, :, :, :]
    print(f"train_point_cloud_sequence shape : {train_point_cloud_sequence.shape}")
    print(f"test_point_cloud_sequence shape : {test_point_cloud_sequence.shape}")
    t_eval = np.array([0, 0.05, 0.1, 0.15, 0.20])
    model, stats = train(
        args=args,
        trainPointCloudSeq=train_point_cloud_sequence,
        testPointCloudSeq=test_point_cloud_sequence,
        t_eval=t_eval
    )
    saveModelStats(model, stats)
