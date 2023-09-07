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
THIS_DIR = os.path.dirname(os.path.abspath(__file__))+'/data' 

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


def train(args, trainDataset, testDataset, trainPointCloudSeq, testPointCloudSeq, t_eval):
    """
    trainDataset: [num_trajs, samples_per_control, num_controls, 20]  # 20 = p (3) + R(12) + v(3) + w(3) + u(2)
    trainPointCloudSeq: [num_trajs, samples_per_control, num_controls, 2, 5, P]  # last two for controls
    testPointCloudSeq: [num_trajs, samples_per_control, num_controls, 2, 5, P]  # last two for controls
    """
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    sim_freq = args.control_freq_hz * args.num_points

    print(f'DEVICE: {device}')

    num_actions = trainDataset.shape[2]
    print(f'num_actions = {num_actions}')

    x_train = trainDataset
    x_train_cat = x_train
    x_train_cat[:, 1:, :, :3] = x_train_cat[:, 1:, :, :3] - np.expand_dims(x_train_cat[:, 0, :, :3], axis=1)
    x_train_cat[:, 0, :, :3] = 0
    x_train_cat = torch.tensor(x_train_cat, requires_grad=True, dtype=torch.float32).to(device)
    x_test = testDataset
    x_test_cat = x_test
    x_test_cat[:, 1:, :, :3] = x_test_cat[:, 1:, :, :3] - np.expand_dims(x_test_cat[:, 0, :, :3], axis=1)
    x_test_cat[:, 0, :, :3] = 0
    x_test_cat = torch.tensor(x_test_cat, requires_grad=True, dtype=torch.float32).to(device)
    # t_eval = np.linspace(0, args.num_points + 1, args.num_points) * 0.05
    t_eval = torch.tensor(t_eval, requires_grad=True, dtype=torch.float32).to(device)
    
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

    # Load saved params if needed
    # path = '{}/PointCloudTrainedModels/jackal-se3ham_pointclouds-rk4-5p-200.tar'.format(args.save_dir)
    # model.load_state_dict(torch.load(path, map_location=device))

    # path = f'{THIS_DIR}/PointCloudTrainedModels/jackal-se3ham_pointclouds-rk4-5p-stats.pkl'
    # stats = from_pickle(path)
    for step in tqdm(range(args.total_steps + 1)):
        if step % 100 == 0 and step > 0:
            for gr in optim.param_groups:
                gr['lr'] *= 0.5
        train_loss = 0
        test_loss = 0

        # Train dataset 
        target_hat = None
        for u in range(trainDataset.shape[2]):
            if u == 0:
                y_pred = odeint_adjoint(
                    model,
                    x_train_cat[:, 0, 0, :],
                    t_eval,
                    method=args.solver
                )  # (5, 18, 24)
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = y_pred
            else:
                y0 = torch.cat((target_hat[:, -1, u-1, 0:18], x_train_cat[:, 0, u, 18:20]), dim=1)
                y_pred = odeint_adjoint(model, y0, t_eval, method=args.solver)
                y_pred = y_pred[:, :, :]
                y_pred = y_pred.permute(1, 0, 2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = torch.cat((target_hat, y_pred), dim=2)
        target_hat = target_hat.permute(1, 0, 2, 3)
        target_hat = target_hat.flatten(1, 2)
        target = pc_train_cat.permute(1, 0, 2, 3, 4, 5)  # pc_train_cat --> 8 X 4 X 60 X 2 X 5 X 10, after permute --> 4 X 8 X 60 X 2 X 5 X 10
        target = target.flatten(1, 2)  # 4 X 480 X 2 X 5 X 10
        
        train_loss = point_cloud_l2_loss_for_successive_point_clouds(
            observedPointClouds=target,
            predictedStates=target_hat,
            split=[model.xdim, model.Rdim, model.twistdim, model.udim]
        )

        gnet_l1_loss = 0
        lambda_gnet_l1_loss = 1e-3
        rand_sample_indx = np.random.randint(0, 5)
        rand_control_indx = np.random.randint(1, target_hat.shape[1])
        q = target_hat[rand_sample_indx, rand_control_indx, :12]
        gnet_l1_loss += torch.abs(model.g_net(q)).mean()
        train_loss += (lambda_gnet_l1_loss * gnet_l1_loss)

        # Test dataset
        target_hat = None
        for u in range(testDataset.shape[2]):
            if u == 0:
                y_pred = odeint_adjoint(model, x_test_cat[:,0,0,:], t_eval, method=args.solver) # (5, 18, 24)
                y_pred = y_pred.permute(1,0,2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat = y_pred
            else:
                y0 = torch.cat((target_hat[:,-1,u-1,0:18], x_test_cat[:,0,u,18:20]), dim=1)
                y_pred = odeint_adjoint(model, y0, t_eval, method=args.solver)
                y_pred = y_pred[:,:,:]
                y_pred = y_pred.permute(1,0,2)
                y_pred = torch.unsqueeze(y_pred, dim=2)
                target_hat= torch.cat((target_hat, y_pred), dim=2)
        target_hat = target_hat.permute(1, 0, 2, 3)
        target_hat = target_hat.flatten(1, 2)
        target = pc_test_cat.permute(1, 0, 2, 3, 4, 5)
        target = target.flatten(1, 2)
    
        test_loss = point_cloud_l2_loss_for_successive_point_clouds(
            observedPointClouds=target,
            predictedStates=target_hat,
            split=[model.xdim, model.Rdim, model.twistdim, model.udim]
        )
        test_loss += (lambda_gnet_l1_loss * gnet_l1_loss)
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
            path = '{}/PointCloudTrainedModels/{}{}-{}-{}p-{}.tar'.format(args.save_dir, args.name, label, args.solver, args.num_points, step)
            torch.save(model.state_dict(), path)

    # After Training 
    x_train = torch.tensor(x_train, requires_grad=True, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float32).to(device)

    train_loss = []
    test_loss = []
    train_l2_loss = []
    test_l2_loss = []
    train_geo_loss = []
    test_geo_loss = []
    train_data_hat = []
    test_data_hat = []
    for i in range(x_train.shape[0]):
        train_x_hat = odeint_adjoint(model, x_train[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = \
            traj_pose_L2_geodesic_loss(x_train[i, :, :, :], train_x_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        train_loss.append(total_loss)
        train_l2_loss.append(l2_loss)
        train_geo_loss.append(geo_loss)
        train_data_hat.append(train_x_hat.detach().cpu().numpy())

    for i in range(x_test.shape[0]):
        # Run test data
        test_x_hat = odeint(model, x_test[i, 0, :, :], t_eval, method=args.solver)
        total_loss, l2_loss, geo_loss = \
            traj_pose_L2_geodesic_loss(x_test[i,:,:,:], test_x_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim])
        test_loss.append(total_loss)
        test_l2_loss.append(l2_loss)
        test_geo_loss.append(geo_loss)
        test_data_hat.append(test_x_hat.detach().cpu().numpy())

    # Stats 
    stats['train_x'] = x_train.detach().cpu().numpy()
    stats['test_x'] = x_test.detach().cpu().numpy()
    stats['train_x_hat'] = np.array(train_data_hat)
    stats['test_x_hat'] = np.array(test_data_hat)
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


def testPlot(dataset, pose):
    num_traj = dataset.shape[0]
    pdf = PdfPages("./img/damp_plots.pdf")
    
    for t in range(num_traj):
        traj = t
        samples = dataset.shape[1]
        trajectory = np.zeros((dataset.shape[1]*dataset.shape[2], 24))
        estTrajectory = np.zeros((dataset.shape[1]*dataset.shape[2], 18))
        rpys = np.zeros((dataset.shape[1]*dataset.shape[2], 3))
        estRpys = np.zeros((dataset.shape[1]*dataset.shape[2], 3))
        for i in range(dataset.shape[2]):
            for j in range(dataset.shape[1]):
                trajectory[samples*i + j, :] = dataset[traj, j, i, :]
                estTrajectory[samples*i + j, :] = pose[traj, j, i, :]
                
                R = trajectory[samples*i + j, 3:12].reshape(3,3)
                rotation = Rotation.from_matrix(R)
                rpys[samples*i + j, :] = rotation.as_euler('xyz')
                
                R = estTrajectory[samples*i + j, 3:12].reshape(3,3)
                rotation = Rotation.from_matrix(R)
                estRpys[samples*i + j, :] = rotation.as_euler('xyz')

        t_eval = np.arange(0, dataset.shape[1]*dataset.shape[2])

        # Figure 
        figsize = (24, 18)
        fontsize = 24
        fontsize_ticks = 32
        line_width = 1

        # Plot 
        fig, axs = plt.subplots(6, 2, figsize=figsize)
        axs[0,0].plot(t_eval, estTrajectory[:,0], 'b', linewidth=line_width, label='output')
        axs[0,0].plot(t_eval, trajectory[:,0], 'r--', linewidth=line_width, label='reference')
        axs[0,0].legend(loc="upper right")
        axs[0,0].set(ylabel=r'$x(t)$')
        axs[0,0].set(title='Position')
        axs[0,0].set_ylim([-2, 2])
        axs[1,0].plot(t_eval, estTrajectory[:,1], 'b', linewidth=line_width, label='output')
        axs[1,0].plot(t_eval, trajectory[:,1], 'r--', linewidth=line_width, label='reference')
        axs[1,0].legend(loc="upper right")
        axs[1,0].set(ylabel=r'$y(t)$')
        axs[1,0].set_ylim([-2, 2])
        axs[2,0].plot(t_eval, estTrajectory[:,2], 'b', linewidth=line_width, label='output')
        axs[2,0].plot(t_eval, trajectory[:,2], 'r--', linewidth=line_width, label='reference')
        axs[2,0].legend(loc="upper right")
        axs[2,0].set(ylabel=r'$z(t)$')
        axs[2,0].set_ylim([-2, 2])
        axs[0,1].plot(t_eval, estRpys[:,0], 'b', linewidth=line_width, label='output')
        axs[0,1].plot(t_eval, rpys[:,0], 'r--', linewidth=line_width, label='reference')
        axs[0,1].legend(loc="upper right")
        axs[0,1].set(ylabel=r'$\phi(t)$')
        axs[0,1].set(title='Angles')
        axs[0,1].set_ylim([-1, 1])
        axs[1,1].plot(t_eval, estRpys[:,1], 'b', linewidth=line_width, label='output')
        axs[1,1].plot(t_eval, rpys[:,1], 'r--', linewidth=line_width, label='reference')
        axs[1,1].legend(loc="upper right")
        axs[1,1].set(ylabel=r'$\theta(t)$')
        axs[1,1].set_ylim([-1, 1])
        axs[2,1].plot(t_eval, estRpys[:,2], 'b', linewidth=line_width, label='output')
        axs[2,1].plot(t_eval, rpys[:,2], 'r--', linewidth=line_width, label='reference')
        axs[2,1].legend(loc="upper right")
        axs[2,1].set(ylabel=r'$\psi(t)$')
        axs[2,1].set_ylim([-1, 1])
        fig.tight_layout()
        axs[3,0].plot(t_eval, estTrajectory[:,12], 'b', linewidth=line_width, label='output')
        axs[3,0].plot(t_eval, trajectory[:,12], 'r--', linewidth=line_width, label='reference')
        axs[3,0].legend(loc="upper right")
        axs[3,0].set(ylabel=r'$v_x(t)$')
        axs[3,0].set(title='Velocity')
        axs[3,0].set_ylim([-2, 2])
        axs[4,0].plot(t_eval, estTrajectory[:,13], 'b', linewidth=line_width, label='output')
        axs[4,0].plot(t_eval, trajectory[:,13], 'r--', linewidth=line_width, label='reference')
        axs[4,0].legend(loc="upper right")
        axs[4,0].set(ylabel=r'$v_y(t)$')
        axs[4,0].set_ylim([-2, 2])
        axs[5,0].plot(t_eval, estTrajectory[:,14], 'b', linewidth=line_width, label='output')
        axs[5,0].plot(t_eval, trajectory[:,14], 'r--', linewidth=line_width, label='reference')
        axs[5,0].legend(loc="upper right")
        axs[5,0].set(ylabel=r'$v_z(t)$')
        axs[5,0].set_ylim([-2, 2])
        axs[3,1].plot(t_eval, estTrajectory[:,15], 'b', linewidth=line_width, label='output')
        axs[3,1].plot(t_eval, trajectory[:,15], 'r--', linewidth=line_width, label='reference')
        axs[3,1].legend(loc="upper right")
        axs[3,1].set(ylabel=r'$\omega_x(t)$')
        axs[3,1].set(title='Angular Velocity')
        axs[3,1].set_ylim([-2, 2])
        axs[4,1].plot(t_eval, estTrajectory[:,16], 'b', linewidth=line_width, label='output')
        axs[4,1].plot(t_eval, trajectory[:,16], 'r--', linewidth=line_width, label='reference')
        axs[4,1].legend(loc="upper right")
        axs[4,1].set(ylabel=r'$\omega_y(t)$')
        axs[4,1].set_ylim([-2, 2])
        axs[5,1].plot(t_eval, estTrajectory[:,17], 'b', linewidth=line_width, label='output')
        axs[5,1].plot(t_eval, trajectory[:,17], 'r--', linewidth=line_width, label='reference')
        axs[5,1].legend(loc="upper right")
        axs[5,1].set(ylabel=r'$\omega_z(t)$')
        axs[5,1].set_ylim([-2, 2])
        plt.xlabel("time")
        plt.title('Trajectory ({})'.format(t))
        # plt.savefig('./img/damp_test_position.pdf', bbox_inches='tight', pad_inches=0.1)
        plt.savefig(pdf, format='pdf') 
    pdf.close()


def loadModelStats():
    stats = {'train_loss': [], 'test_loss': [], 'train_x': [], 'test_x': [], 'train_x_hat': [], 'test_x_hat': [], 't_eval': []}
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    model =  SE3HamNODE(device=device, pretrain = True).to(device)
    path = 'data/rigidBody-se3ham-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = 'data/rigidBody-se3ham-rk4-5p-stats.pkl'
    with open(path, 'rb') as handle:
        stats = pickle.load(handle)
    return model, stats


if __name__ == "__main__":
    args = get_args()
    data = get_dataset(test_split=0.8, save_dir=args.save_dir)
    train_x, t_eval = arrange_data(data['x'], data['t'], num_points=args.num_points)
    test_x, _ = arrange_data(data['test_x'], data['t'], num_points=args.num_points)
    print(f"train_x shape : {train_x.shape}")
    print(f"test_x shape : {test_x.shape}")
    point_cloud_data = np.load(f"{THIS_DIR}/PointCloudData.npy")
    num_controls = point_cloud_data.shape[2]
    split_idx = int(num_controls * 0.8)
    train_point_cloud_sequence = point_cloud_data[:, :, :split_idx, :, :, :]
    test_point_cloud_sequence = point_cloud_data[:, :, split_idx:, :, :, :]
    print(f"train_point_cloud_sequence shape : {train_point_cloud_sequence.shape}")
    print(f"test_point_cloud_sequence shape : {test_point_cloud_sequence.shape}")
    model, stats = train(
        args=args,
        trainDataset=train_x,
        testDataset=test_x,
        trainPointCloudSeq=train_point_cloud_sequence,
        testPointCloudSeq=test_point_cloud_sequence,
        t_eval=t_eval
    )
    saveModelStats(model, stats)
