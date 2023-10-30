import torch, os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp
from se3hamneuralode import from_pickle, SE3HamNODE

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data/'
gpu = 0
device = torch.device('cuda:' + str(gpu) if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--dt', default='0.1', type=float, help='sampling time')
    parser.add_argument('--filter', default='0', type=int, help='filtered linear velocity and angular velocity')
    parser.add_argument('--name', default='jackal', type=str, help='only one option right now')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--solver', default='rk4', type=str, help='type of ODE Solver for Neural ODE')
    parser.set_defaults(feature=True)
    args, unknown = parser.parse_known_args()  # use parse_known_args instead of parse_args
    return args


def get_model(M1_known, M2_known):
    model = SE3HamNODE(device=device, pretrain=False, M1_known=M1_known, M2_known=M2_known).to(device)
    path = f'{THIS_DIR}PointCloudTrainedModels/TrainedModel/jackal-se3ham_pointclouds-rk4-5p.tar'
    model.load_state_dict(torch.load(path, map_location=device))
    path = f'{THIS_DIR}PointCloudTrainedModels/TrainedModel/jackal-se3ham_pointclouds-rk4-5p-stats.pkl'
    stats = from_pickle(path)
    return model, stats


if __name__ == "__main__":
    M1_known, M2_known = False, False
    # Load trained model
    model, stats = get_model(M1_known, M2_known)
    args = get_args()
    # Load train/test data
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']
    print("Loaded data!")

    # Pick a sample test trajectory
    traj = 0
    sample_traj = test_x_hat[:, traj, 0:18]
    num_samples = sample_traj.shape[0]

    # This is the generalized coordinates q = pose
    qp = torch.tensor(sample_traj[:, :], requires_grad=True, dtype=torch.float32).to(device)
    x_pv = torch.cat((qp[:, 0:3], qp[:, 12:15]), dim=1)
    R_pw = torch.cat((qp[:, 3:12], qp[:, 15:18]), dim=1)
    pose = torch.tensor(sample_traj[:, 0:12], requires_grad=True, dtype=torch.float32).to(device)
    x, R = torch.split(pose, [3, 9], dim=1)

    # Plot loss
    figsize = (12, 7.8)
    fontsize = 30
    fontsize_ticks = 38
    line_width = 4
    framealpha = 0.05
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['text.usetex'] = True
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    iterations = len(train_loss)
    plt.figure(figsize=figsize)
    plt.plot(train_loss, 'b', linewidth=line_width, label='train loss')
    plt.plot(test_loss, 'r--', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.yscale('log')
    plt.xticks(fontsize=fontsize_ticks)
    plt.legend(loc="upper right", fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/loss_log.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    # Show a grid for each log y-axis tick
    plt.grid(which='both')
    plt.show(block=True)

    # Load train/test data
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']
    print("Loaded data!")

    # Pick a sample test trajectory
    traj = 0
    sample_traj = test_x_hat[:, traj, 0:18]
    num_samples = sample_traj.shape[0]

    # This is the generalized coordinates q = pose
    qp = torch.tensor(sample_traj[:, :], requires_grad=True, dtype=torch.float32).to(device)
    x_pv = torch.cat((qp[:, 0:3], qp[:, 12:15]), dim=1)
    R_pw = torch.cat((qp[:, 3:12], qp[:, 15:18]), dim=1)
    pose = torch.tensor(sample_traj[:, 0:12], requires_grad=True, dtype=torch.float32).to(device)
    x, R = torch.split(pose, [3, 9], dim=1)

    # Calculate the M^-1, V, g for the q.
    M_q_inv1 = model.M_net1(x)
    M_q_inv2 = model.M_net2(R)
    g_q = model.g_net(pose)
    D_v = model.D_net(qp)[..., :3, :3]
    D_w = model.D_net(qp)[..., 3:, 3:]
    V_q = model.V_net(pose)
    timestamp = np.arange(0, sample_traj.shape[0] * args.dt - 1e-10, args.dt)

    det = []
    RRT_I_dist = []
    for i in range(len(R)):
        R_hat = R[i].detach().cpu().numpy().reshape(3, 3)
        R_det = np.linalg.det(R_hat)
        det.append(np.abs(R_det - 1))

        R_RT = R_hat @ R_hat.T
        RRT_I = np.linalg.norm(R_RT - np.eye(3))
        RRT_I_dist.append(RRT_I)

    plt.figure(figsize=figsize)
    plt.plot(timestamp, det, 'b', linewidth=line_width, label=r'$|det(R) - 1|$')
    plt.plot(timestamp, RRT_I_dist, 'r', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$')
    plt.xlabel("t", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    y_tick_positions = [8e-7, 6e-7, 4e-7, 2e-7]  # Adjust these to your desired positions
    y_tick_labels = ['$8e-07$', '$6e-07$', '$4e-07$', '$2e-07$']  # Labels corresponding to the positions in LaTeX format
    plt.yticks(y_tick_positions, y_tick_labels, fontsize=fontsize_ticks)
    plt.legend(loc="best", fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/SO3_constraints_test.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)

    # Plot V(q)
    plt.figure(figsize=figsize)
    plt.plot(timestamp, V_q.detach().cpu().numpy(), 'tab:blue', linestyle='dashed', label=r'$V(q)$', linewidth=3)
    plt.xlabel("$z$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/V_x.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)

    # Plot M1^-1(q)
    plt.figure(figsize=figsize)
    plt.plot(timestamp, M_q_inv1.detach().cpu().numpy()[:, 0, 0], 'tab:orange', linestyle='dashed',
             linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[0,0]$')
    plt.plot(timestamp, M_q_inv1.detach().cpu().numpy()[:, 1, 1], 'tab:blue', linestyle='dashed', linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[1,1]$')
    plt.plot(timestamp, M_q_inv1.detach().cpu().numpy()[:, 2, 2], 'tab:red', linestyle='dashed', linewidth=line_width,
             alpha=0.2,
             label=r'$M^{-1}_{1}(q)[2,2]$')
    plt.plot(timestamp, M_q_inv1.detach().cpu().numpy()[:, 1, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2, label=r'$M^{-1}_{1}(q)[1,0]$')
    plt.plot(timestamp, M_q_inv1.detach().cpu().numpy()[:, 2, 0], 'tab:gray', linestyle='dashed',
             linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[2,0]$')
    plt.plot(timestamp, M_q_inv1.detach().cpu().numpy()[:, 2, 1], 'tab:gray', linestyle='dashed',
             linewidth=line_width,
             label=r'$M^{-1}_{1}(q)[2,1]$')
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(loc="lower right", fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/M1_x_all.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)

    # Plot M2^-1(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 0, 0], 'tab:orange', linestyle='dashed',
             linewidth=line_width,
             label=r'$M^{-1}_{2}(q)[0, 0]$')
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 1, 1], 'tab:green', linestyle='dashed', linewidth=line_width,
             label=r'$M^{-1}_{2}(q)[1, 1]$')
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 2, 2], 'tab:blue', linestyle='dashed', linewidth=line_width,
             label=r'$M^{-1}_{2}(q)[2,2]$')
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 0, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2,
             label=r'Other $M^{-1}_{2}(q)[i,j]$')
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 0, 2], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 1, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 1, 2], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 2, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, M_q_inv2.detach().cpu().numpy()[:, 2, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize, loc='lower right')
    plt.savefig(f'{THIS_DIR}png/M2_x_all.pdf', format='pdf', bbox_inches='tight')
    plt.show(block=True)

    # Plot Dv(q)
    plt.figure(figsize=figsize)
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 0, 0], 'tab:orange', linestyle='dashed', linewidth=line_width,
             label=r'$D_{v}(q)[0,0]$')
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 1, 1], 'tab:blue', linestyle='dashed', linewidth=line_width,
             label=r'$D_{v}(q)[1,1]$')
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 2, 2], 'tab:green', linestyle='dashed', linewidth=line_width,
             label=r'$D_{v}(q)[2,2]$')
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 0, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             label=r'Other $D_{v}(q)[i,j]$')
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 0, 2], 'tab:gray', linestyle='dashed',
             linewidth=line_width, alpha=0.5)
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 1, 0], 'tab:gray', linestyle='dashed', linewidth=line_width)

    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 1, 2], 'tab:gray', linestyle='dashed', linewidth=line_width, alpha=0.2)
    plt.plot(timestamp, D_v.detach().cpu().numpy()[:, 2, 0], 'tab:gray', linestyle='dashed', linewidth=line_width)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(loc="upper right", fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/Dv_x_all.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)

    # Plot Dw(q)
    plt.figure(figsize=figsize)
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 0, 0], 'tab:orange', linestyle='dashed', linewidth=line_width,
             label=r'$D_{w}(q)[0,0]$')
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 1, 1], 'tab:blue', linestyle='dashed', linewidth=line_width,
             label=r'$D_{w}(q)[1,1]$')
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 2, 2], 'tab:green', linestyle='dashed', linewidth=line_width,
             label=r'$D_{w}(q)[2,2]$')
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 0, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             label=r'Other $D_{w}(q)[i,j]$')
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 0, 2], 'tab:gray', linestyle='dashed', linewidth=line_width)
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 1, 0], 'tab:gray', linestyle='dashed', linewidth=line_width)

    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 1, 2], 'tab:gray', linestyle='dashed', linewidth=line_width)
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 2, 0], 'tab:gray', linestyle='dashed', linewidth=line_width)
    plt.plot(timestamp, D_w.detach().cpu().numpy()[:, 2, 1], 'tab:gray', linestyle='dashed', linewidth=line_width)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(loc="upper right", fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/Dw_x_all.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)

    # Plot g_(q)
    plt.figure(figsize=figsize)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 0, 0], 'tab:blue', linestyle='dashed', linewidth=line_width,
             label=r'$g(q)[0,0]$')
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 0, 1], 'tab:orange', linestyle='dashed', linewidth=line_width,
             label=r'$g(q)[0,1]$')
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 5, 0], 'tab:green', linestyle='dashed', linewidth=line_width,
             label=r'$g(q)[5,0]$')
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 5, 1], 'tab:red', linestyle='dashed', linewidth=line_width,
             label=r'$g(q)[5,1]$')
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 1, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2,
             label=r'Other $g(q)$')
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 1, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 2, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 2, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 3, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 3, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 4, 0], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.plot(timestamp, g_q.detach().cpu().numpy()[:, 4, 1], 'tab:gray', linestyle='dashed', linewidth=line_width,
             alpha=0.2)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(loc="lower right", fontsize=fontsize, framealpha=framealpha)
    plt.savefig(f'{THIS_DIR}png/g_x.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show(block=True)

    # np.save("train_loss.npy", train_loss)
    # np.save("test_loss.npy", test_loss)
    # np.save("V_q.npy", V_q.detach().cpu().numpy())
    # np.save("M1_inverse.npy", M_q_inv1.detach().cpu().numpy())
    # np.save("M2_inverse.npy", M_q_inv2.detach().cpu().numpy())
    # np.save("D_v.npy", D_v.detach().cpu().numpy())
    # np.save("D_w.npy", D_w.detach().cpu().numpy())
    # np.save("g.npy", g_q.detach().cpu().numpy())
    # np.save("SO3_determinant.npy", np.array(det))
    # np.save("SO3_RRT_I_dist.npy", np.array(RRT_I_dist))