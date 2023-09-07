import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.transform import Rotation


def plot_dataset(outTraj, refTraj, sim_freq=240.0):
    num_traj = outTraj.shape[0]
    samples_per_control = outTraj.shape[1]
    samples = outTraj.shape[2]

    # Trajectory 
    outTrajectory = np.zeros((num_traj, samples_per_control*samples, 24))
    refTrajectory = np.zeros((num_traj, samples_per_control*samples, 18))
    outRpys = np.zeros((num_traj, samples_per_control*samples, 3))
    refRpys = np.zeros((num_traj, samples_per_control*samples, 3))

    for t in range(num_traj):
        # Output 
        for i in range(samples):
            for j in range(samples_per_control):
                outTrajectory[t, samples_per_control*i + j, :] = outTraj[t, j, i, :]
                R = outTrajectory[t, samples_per_control*i + j, 3:12].reshape(3,3)
                rotation = Rotation.from_matrix(R)
                outRpys[t, samples_per_control*i + j, :] = rotation.as_euler('xyz') 

                refTrajectory[t, samples_per_control*i + j, :] = refTraj[t,:]
                R = refTraj[t, 3:12].reshape(3,3)
                rotation = Rotation.from_matrix(R)
                refRpys[t, samples_per_control*i + j, :] = rotation.as_euler('xyz')
    
    # Time Evaluation 
    t_eval = np.arange(0.0, outTrajectory.shape[1]/sim_freq, 1/sim_freq)

    # Plot 
    pdf = PdfPages("./img/dataset_plot.pdf")
    figsize = (24, 18)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 1

    for t in range(num_traj):
        fig, axs = plt.subplots(6, 2, figsize=figsize)
        axs[0,0].plot(t_eval, outTrajectory[t,:,0], 'b', linewidth=line_width, label='output')
        axs[0,0].plot(t_eval, refTrajectory[t,:,0], 'r--', linewidth=line_width, label='reference')
        axs[0,0].legend(loc="upper right")
        axs[0,0].set(ylabel=r'$x(t)$')
        axs[0,0].set(title='Position')
        axs[0,0].set_ylim([-2, 2])
        axs[1,0].plot(t_eval, outTrajectory[t,:,1], 'b', linewidth=line_width, label='output')
        axs[1,0].plot(t_eval, refTrajectory[t,:,1], 'r--', linewidth=line_width, label='reference')
        axs[1,0].legend(loc="upper right")
        axs[1,0].set(ylabel=r'$y(t)$')
        axs[1,0].set_ylim([-2, 2])
        axs[2,0].plot(t_eval, outTrajectory[t,:,2], 'b', linewidth=line_width, label='output')
        axs[2,0].plot(t_eval, refTrajectory[t,:,2], 'r--', linewidth=line_width, label='reference')
        axs[2,0].legend(loc="upper right")
        axs[2,0].set(ylabel=r'$z(t)$')
        axs[2,0].set_ylim([-2, 2])
        axs[0,1].plot(t_eval, outRpys[t,:,0], 'b', linewidth=line_width, label='output')
        axs[0,1].plot(t_eval, refRpys[t,:,0], 'r--', linewidth=line_width, label='reference')
        axs[0,1].legend(loc="upper right")
        axs[0,1].set(ylabel=r'$\phi(t)$')
        axs[0,1].set(title='Angles')
        axs[0,1].set_ylim([-1, 1])
        axs[1,1].plot(t_eval, outRpys[t,:,1], 'b', linewidth=line_width, label='output')
        axs[1,1].plot(t_eval, refRpys[t,:,1], 'r--', linewidth=line_width, label='reference')
        axs[1,1].legend(loc="upper right")
        axs[1,1].set(ylabel=r'$\theta(t)$')
        axs[1,1].set_ylim([-1, 1])
        axs[2,1].plot(t_eval, outRpys[t,:,2], 'b', linewidth=line_width, label='output')
        axs[2,1].plot(t_eval, refRpys[t,:,2], 'r--', linewidth=line_width, label='reference')
        axs[2,1].legend(loc="upper right")
        axs[2,1].set(ylabel=r'$\psi(t)$')
        axs[2,1].set_ylim([-1, 1])
        fig.tight_layout()
        axs[3,0].plot(t_eval, outTrajectory[t,:,12], 'b', linewidth=line_width, label='output')
        axs[3,0].plot(t_eval, refTrajectory[t,:,12], 'r--', linewidth=line_width, label='reference')
        axs[3,0].legend(loc="upper right")
        axs[3,0].set(ylabel=r'$v_x(t)$')
        axs[3,0].set(title='Velocity')
        axs[3,0].set_ylim([-2, 2])
        axs[4,0].plot(t_eval, outTrajectory[t,:,13], 'b', linewidth=line_width, label='output')
        axs[4,0].plot(t_eval, refTrajectory[t,:,13], 'r--', linewidth=line_width, label='reference')
        axs[4,0].legend(loc="upper right")
        axs[4,0].set(ylabel=r'$v_y(t)$')
        axs[4,0].set_ylim([-2, 2])
        axs[5,0].plot(t_eval, outTrajectory[t,:,14], 'b', linewidth=line_width, label='output')
        axs[5,0].plot(t_eval, refTrajectory[t,:,14], 'r--', linewidth=line_width, label='reference')
        axs[5,0].legend(loc="upper right")
        axs[5,0].set(ylabel=r'$v_z(t)$')
        axs[5,0].set_ylim([-2, 2])
        axs[3,1].plot(t_eval, outTrajectory[t,:,15], 'b', linewidth=line_width, label='output')
        axs[3,1].plot(t_eval, refTrajectory[t,:,15], 'r--', linewidth=line_width, label='reference')
        axs[3,1].legend(loc="upper right")
        axs[3,1].set(ylabel=r'$\omega_x(t)$')
        axs[3,1].set(title='Angular Velocity')
        axs[3,1].set_ylim([-2, 2])
        axs[4,1].plot(t_eval, outTrajectory[t,:,16], 'b', linewidth=line_width, label='output')
        axs[4,1].plot(t_eval, refTrajectory[t,:,16], 'r--', linewidth=line_width, label='reference')
        axs[4,1].legend(loc="upper right")
        axs[4,1].set(ylabel=r'$\omega_y(t)$')
        axs[4,1].set_ylim([-2, 2])
        axs[5,1].plot(t_eval, outTrajectory[t,:,17], 'b', linewidth=line_width, label='output')
        axs[5,1].plot(t_eval, refTrajectory[t,:,17], 'r--', linewidth=line_width, label='reference')
        axs[5,1].legend(loc="upper right")
        axs[5,1].set(ylabel=r'$\omega_z(t)$')
        axs[5,1].set_ylim([-2, 2])
        plt.xlabel("time")
        plt.title('Trajectory ({})'.format(t))
        plt.savefig(pdf, format='pdf') 
    pdf.close()

def plot_trajectory(traj, refTraj, sim_freq=240.0):
    samples = traj.shape[0]
    trajectory = np.zeros((traj.shape[0]*traj.shape[1], 24))
    rpys = np.zeros((traj.shape[0]*traj.shape[1], 3))
    for i in range(traj.shape[1]):
        for j in range(traj.shape[0]):
            trajectory[samples*i + j, :] = traj[j, i, :]
            R = trajectory[samples*i + j, 3:12].reshape(3,3)
            rotation = Rotation.from_matrix(R)
            rpys[samples*i + j, :] = rotation.as_euler('xyz') 

    refTrajectory = np.repeat(refTraj, traj.shape[0]*traj.shape[1]).reshape(-1, traj.shape[0]*traj.shape[1])
    t_eval = np.arange(0.0, trajectory.shape[0]/sim_freq, 1/sim_freq)

    # Ref
    R = refTraj[3:12].reshape(3,3)
    rotation = Rotation.from_matrix(R)
    refRpys = np.repeat(rotation.as_euler('xyz'), traj.shape[0]*traj.shape[1]).reshape(-1, traj.shape[0]*traj.shape[1])

    print(f'traj = {traj.shape}')

    figsize = (24, 18)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 1

    # Plot 
    fig, axs = plt.subplots(9, 2, figsize=figsize)
    axs[0,0].plot(t_eval, trajectory[:,0], 'b', linewidth=line_width, label='output')
    axs[0,0].plot(t_eval, refTrajectory[0,:], 'r', linewidth=line_width, label='reference')
    axs[0,0].legend(loc="upper right")
    axs[0,0].set(ylabel=r'$x(t)$')
    axs[0,0].set(title='Position')
    axs[0,0].set_ylim([-2, 2])
    axs[1,0].plot(t_eval, trajectory[:,1], 'b', linewidth=line_width, label='output')
    axs[1,0].plot(t_eval, refTrajectory[1,:], 'r', linewidth=line_width, label='reference')
    axs[1,0].legend(loc="upper right")
    axs[1,0].set(ylabel=r'$y(t)$')
    axs[1,0].set_ylim([-2, 2])
    axs[2,0].plot(t_eval, trajectory[:,2], 'b', linewidth=line_width, label='output')
    axs[2,0].plot(t_eval, refTrajectory[2,:], 'r', linewidth=line_width, label='reference')
    axs[2,0].legend(loc="upper right")
    axs[2,0].set(ylabel=r'$z(t)$')
    axs[2,0].set_ylim([-2, 2])

    axs[0,1].plot(t_eval, rpys[:,0], 'b', linewidth=line_width, label='output')
    axs[0,1].plot(t_eval, refRpys[0,:], 'r', linewidth=line_width, label='reference')
    axs[0,1].legend(loc="upper right")
    axs[0,1].set(ylabel=r'$\phi(t)$')
    axs[0,1].set(title='Angles')
    axs[0,1].set_ylim([-2, 2])
    axs[1,1].plot(t_eval, rpys[:,1], 'b', linewidth=line_width, label='output')
    axs[1,1].plot(t_eval, refRpys[1,:], 'r', linewidth=line_width, label='reference')
    axs[1,1].legend(loc="upper right")
    axs[1,1].set(ylabel=r'$\theta(t)$')
    axs[1,1].set_ylim([-2, 2])
    axs[2,1].plot(t_eval, rpys[:,2], 'b', linewidth=line_width, label='output')
    axs[2,1].plot(t_eval, refRpys[2,:], 'r', linewidth=line_width, label='reference')
    axs[2,1].legend(loc="upper right")
    axs[2,1].set(ylabel=r'$\psi(t)$')
    axs[2,1].set_ylim([-2, 2])

    axs[3,0].plot(t_eval, trajectory[:,12], 'b', linewidth=line_width, label='output')
    axs[3,0].plot(t_eval, refTrajectory[12,:], 'r', linewidth=line_width, label='reference')
    axs[3,0].legend(loc="upper right")
    axs[3,0].set(ylabel=r'$v_x(t)$')
    axs[3,0].set(title='Velocity')
    axs[3,0].set_ylim([-2, 2])
    axs[4,0].plot(t_eval, trajectory[:,13], 'b', linewidth=line_width, label='output')
    axs[4,0].plot(t_eval, refTrajectory[13,:], 'r', linewidth=line_width, label='reference')
    axs[4,0].legend(loc="upper right")
    axs[4,0].set(ylabel=r'$v_y(t)$')
    axs[4,0].set_ylim([-2, 2])
    axs[5,0].plot(t_eval, trajectory[:,14], 'b', linewidth=line_width, label='output')
    axs[5,0].plot(t_eval, refTrajectory[14,:], 'r', linewidth=line_width, label='reference')
    axs[5,0].legend(loc="upper right")
    axs[5,0].set(ylabel=r'$v_z(t)$')
    axs[5,0].set_ylim([-2, 2])

    axs[3,1].plot(t_eval, trajectory[:,15], 'b', linewidth=line_width, label='output')
    axs[3,1].plot(t_eval, refTrajectory[15,:], 'r', linewidth=line_width, label='reference')
    axs[3,1].legend(loc="upper right")
    axs[3,1].set(ylabel=r'$\omega_x(t)$')
    axs[3,1].set(title='Angular Velocity')
    # axs[3,1].set_ylim([-2, 2])
    axs[4,1].plot(t_eval, trajectory[:,16], 'b', linewidth=line_width, label='output')
    axs[4,1].plot(t_eval, refTrajectory[16,:], 'r', linewidth=line_width, label='reference')
    axs[4,1].legend(loc="upper right")
    axs[4,1].set(ylabel=r'$\omega_y(t)$')
    # axs[4,1].set_ylim([-2, 2])
    axs[5,1].plot(t_eval, trajectory[:,17], 'b', linewidth=line_width, label='output')
    axs[5,1].plot(t_eval, refTrajectory[17,:], 'r', linewidth=line_width, label='reference')
    axs[5,1].legend(loc="upper right")
    axs[5,1].set(ylabel=r'$\omega_z(t)$')
    # axs[5,1].set_ylim([-2, 2])

    axs[6,0].plot(t_eval, trajectory[:,18], 'b', linewidth=line_width, label='output')
    axs[6,0].legend(loc="upper right")
    axs[6,0].set(ylabel=r'$force_x(t)$')
    axs[6,0].set(title='Controls')
    axs[7,0].plot(t_eval, trajectory[:,19], 'b', linewidth=line_width, label='output')
    axs[7,0].legend(loc="upper right")
    axs[7,0].set(ylabel=r'$force_y(t)$')
    axs[8,0].plot(t_eval, trajectory[:,20], 'b', linewidth=line_width, label='output')
    axs[8,0].legend(loc="upper right")
    axs[8,0].set(ylabel=r'$force_z(t)$')

    axs[6,1].plot(t_eval, trajectory[:,21], 'b', linewidth=line_width, label='output')
    axs[6,1].legend(loc="upper right")
    axs[6,1].set(ylabel=r'$\tau_x(t)$')
    axs[6,1].set(title='Controls')
    axs[7,1].plot(t_eval, trajectory[:,22], 'b', linewidth=line_width, label='output')
    axs[7,1].legend(loc="upper right")
    axs[7,1].set(ylabel=r'$\tau_y(t)$')
    axs[8,1].plot(t_eval, trajectory[:,23], 'b', linewidth=line_width, label='output')
    axs[8,1].legend(loc="upper right")
    axs[8,1].set(ylabel=r'$\tau_z(t)$')

    fig.tight_layout()
    plt.xlabel("time")
    plt.savefig('./img/fig.pdf', bbox_inches='tight', pad_inches=0.1)

    

def plot_loss(stats):
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    fig = plt.figure(figsize=figsize)
    train_loss = stats['train_loss']
    test_loss = stats['test_loss']
    iterations = len(train_loss)
    plt.plot(train_loss, 'b', linewidth=line_width, label='train loss')
    plt.plot(test_loss, 'r--', linewidth=line_width, label='test loss')
    plt.xlabel("iterations", fontsize=fontsize_ticks)
    plt.yscale('log')
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./img/loss_log.png', bbox_inches='tight', pad_inches=0.1)

def plot_SE3Constraints(stats):
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4

    # Load 
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']

    # Pick a sample test trajectory
    num = 2
    sample_traj = test_x[:,num,0:12]
    sample_traj_hat = test_x_hat[:,num,0:12]


    det = []
    RRT_I_dist = []
    for i in range(len(sample_traj_hat)):
        R_hat = sample_traj_hat[i,3:12]
        R_hat = R_hat.reshape(3,3)
        R_det = np.linalg.det(R_hat)
        det.append(np.abs(R_det - 1))
        R_RT = np.matmul(R_hat, R_hat.transpose())
        RRT_I = np.linalg.norm(R_RT - np.diag([1.0, 1.0, 1.0]))
        RRT_I_dist.append(RRT_I)

    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, det, 'b', linewidth=line_width, label=r'$|det(R) - 1|$')
    plt.plot(t_eval, RRT_I_dist, 'r', linewidth=line_width, label=r'$\Vert R R^\top - I\Vert$')
    plt.xlabel("t", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./img/SO3_constraints.png', bbox_inches='tight', pad_inches=0.1)


def plot_V(model, stats):
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4

    # Load 
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']
    t_eval = t_eval[0:5]

    # Pick a sample test trajectory
    num = 2
    sample_traj = test_x[num,:,2,0:12]
    sample_traj_hat = test_x_hat[num,:,2,0:12]

    pose = torch.tensor(sample_traj, requires_grad=True, dtype=torch.float32).to(model.device)
    x, R = torch.split(pose, [3, 9], dim=1)

    # Calculate the M^-1, V, g for the q.
    V_q = model.V_net(pose)
    # M_q_inv1 = model.M_net1(x)
    M_q_inv2 = model.M_net2(R)
    g_q = model.g_net(pose)


    # Print 
    print(f'V(q) = {V_q}')


    fig = plt.figure(figsize=figsize)
    temp = pose[:, 2]
    plt.plot(sample_traj[:,2], V_q.detach().cpu().numpy(), 'b--', label=r'$V(q)$', linewidth=3)
    plt.xlabel("$z$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./img/V_x.png', bbox_inches='tight', pad_inches=0.1)

def plot_M(model, stats):
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    alpha = 0.8
    scale_alpha = 0.0
    scale_beta = 0.0

    # Load 
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']

    # Pick a sample test trajectory
    num = 2
    sample_traj = test_x[num,:,2,0:12]
    sample_traj_hat = test_x_hat[num,:,2,0:12]

    pose = torch.tensor(sample_traj, requires_grad=True, dtype=torch.float32).to(model.device)
    x, R = torch.split(pose, [3, 9], dim=1)

    # Calculate the M^-1, V, g for the q.
    V_q = model.V_net(pose)
    # M_q_inv1 = model.M_net1(x)
    M_q_inv2 = model.M_net2(R)
    g_q = model.g_net(pose)


    # Print 
    # print(f'M1_inv = {M_q_inv1}')
    print(f'M2_inv = {M_q_inv2}')


    # # Plot M1^-1(q)
    # fig = plt.figure(figsize=figsize)
    # plt.plot(t_eval, M_q_inv1_gt.detach().cpu().numpy()[:, 0, 0], 'k-', linewidth=line_width, alpha=alpha, label=r'Ground-truth')
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 2, 2], 'r--', linewidth=line_width, alpha=alpha, label=r'$M^{-1}_{1}(q)[0,0]$')
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 0, 0], 'g--', linewidth=line_width, alpha=alpha, label=r'$M^{-1}_{1}(q)[1,1]$')
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 1, 1], 'b--', linewidth=line_width, alpha=alpha, label=r'$M^{-1}_{1}(q)[2,2]$')
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 0, 1], 'c--', linewidth=line_width, alpha=alpha, label=r'Other $M^{-1}_{1}(q)[i,j]$')
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 0, 2], 'c--', linewidth=line_width, alpha=alpha,)
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width, alpha=alpha,)
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 1, 2], 'c--', linewidth=line_width, alpha=alpha,)
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 2, 0], 'c--', linewidth=line_width, alpha=alpha,)
    # plt.plot(t_eval, M_q_inv1.detach().cpu().numpy()[:, 2, 1], 'c--', linewidth=line_width, alpha=alpha,)
    # plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    # plt.xticks(fontsize=fontsize_ticks)
    # plt.yticks(fontsize=fontsize_ticks)
    # plt.legend(fontsize=fontsize)
    # plt.savefig('./img/M1_x_all.png', bbox_inches='tight', pad_inches=0.1)

    # Plot M2^-1(q)
    fig = plt.figure(figsize=figsize)
    t_eval = t_eval[0:5]
    print(f't_eval = {t_eval.shape}')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:,0, 0], 'r--', linewidth=line_width, alpha=alpha,
             label=r'$M^{-1}_{2}(q)[0, 0]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 1, 1],'g--', linewidth=line_width, alpha=alpha,
             label=r'$M^{-1}_{2}(q)[1, 1]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:,2, 2], 'b--',linewidth=line_width, alpha=alpha,
             label=r'$M^{-1}_{2}(q)[2,2]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 0, 1], 'c--', linewidth=line_width, alpha=alpha,
             label=r'Other $M^{-1}_{2}(q)[i,j]$')
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 0, 2], 'c--', linewidth=line_width, alpha=alpha,)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 1, 0], 'c--', linewidth=line_width, alpha=alpha,)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 1, 2], 'c--', linewidth=line_width, alpha=alpha,)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 2, 0], 'c--', linewidth=line_width, alpha=alpha,)
    plt.plot(t_eval, M_q_inv2.detach().cpu().numpy()[:, 2, 1], 'c--', linewidth=line_width, alpha=alpha,)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize, loc = 'lower right')
    plt.savefig('./img/M2_x_all.png', bbox_inches='tight')

def plot_g(model, stats):
    figsize = (12, 7.8)
    fontsize = 24
    fontsize_ticks = 32
    line_width = 4
    alpha = 0.8

    # Load 
    train_x_hat = stats['train_x_hat']
    test_x_hat = stats['test_x_hat']
    train_x = stats['train_x']
    test_x = stats['test_x']
    t_eval = stats['t_eval']
    t_eval = t_eval[0:5]

    # Pick a sample test trajectory
    num = 2
    sample_traj = test_x[num,:,2,0:12]
    sample_traj_hat = test_x_hat[num,:,2,0:12]

    pose = torch.tensor(sample_traj, requires_grad=True, dtype=torch.float32).to(model.device)
    x, R = torch.split(pose, [3, 9], dim=1)

    # Calculate the M^-1, V, g for the q.
    V_q = model.V_net(pose)
    # M_q_inv1 = model.M_net1(x)
    M_q_inv2 = model.M_net2(R)
    g_q = model.g_net(pose)

    print(f'g_q = {g_q}')

    


    # Plot g(q)
    fig = plt.figure(figsize=figsize)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 1], 'g-.', linewidth=1, label=r'Other $g(q)[i,j]$', alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 2], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 3], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 4], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 5], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 0], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 2], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 3], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 4], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 5], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 0], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 1], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 3], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 4], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 5], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 0], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 1], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 2], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 4], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 5], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 0], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 1], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 2], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 3], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 5], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 0], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 1], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 2], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 3], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 4], 'g-.', linewidth=1, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 0, 0], 'b--', linewidth=line_width, label=r'Diag $g(q)[i,i]$', alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 1, 1], 'b--', linewidth=line_width, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 2, 2], 'b--', linewidth=line_width, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 3, 3], 'b--', linewidth=line_width, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 4, 4], 'b--', linewidth=line_width, alpha=alpha)
    plt.plot(t_eval, g_q.detach().cpu().numpy()[:, 5, 5], 'b--', linewidth=line_width, alpha=alpha)
    plt.xlabel("$t(s)$", fontsize=fontsize_ticks)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    plt.legend(fontsize=fontsize)
    plt.savefig('./img/g_x.png', bbox_inches='tight', pad_inches=0.1)

    