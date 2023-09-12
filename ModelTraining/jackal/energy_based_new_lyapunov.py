import os
import numpy as np
from numpy.linalg import norm
from numpy import trace as tr
import torch
from transforms3d.euler import mat2euler, euler2mat
import time
from se3hamneuralode import SE3HamNODE, from_pickle
from scipy.linalg import logm, expm
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) + '/data'


def hat_map(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def vee_map(R):
    return np.array([R[2, 1], R[0, 2], -R[0, 1]]).reshape(3)


class EnergyBasedController:
    """
    Derivation
    """

    def __init__(self, M1_known: bool = False, M2_known: bool = False, maxTorque: float =2, Kp = None, Kv = None, KR1 = None, KR2 = None, Kw = None, checkpoint=-1):
        checkpoint = '' if checkpoint == -1 else ('-' + str(checkpoint))
        self.M1_known = M1_known
        self.M2_known = M2_known
        # Dimensions
        self.posdim = 3
        self.eulerdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.qdim = self.posdim + self.eulerdim + self.linveldim + self.angveldim
        self.udim = 2
        self.vehicle_width = 0.50
        self.wheel_radius = 0.1
        self.model = self.get_model(checkpoint=checkpoint)

        # Parameters
        if self.M1_known:
            self.M = np.eye(3) * 9
        else:
            self.M = np.eye(3)
            self.Mnet = self.model.M_net1

        if self.M2_known:
            self.Inertia = np.diag([0.25, 0.25, 0.33])
        else:
            self.Inertia = np.diag([1, 1, 1])
            self.Inet = self.model.M_net2

        self.gnet = self.model.g_net
        self.Dnet = self.model.D_net
        self.maxTorque = maxTorque

        self.wheel_radius2 = self.wheel_radius * self.wheel_radius
        self.wheel_distance = 0.5
        self.wheel_distance2 = self.wheel_distance * self.wheel_distance
        self.terrain_param = 1

        self.mass = 8.5
        self.Jzz = 0.33
        self.A = np.array([[self.wheel_radius / 2, self.wheel_radius / 2],
                           [-self.wheel_radius / (self.terrain_param * self.wheel_distance),
                            self.wheel_radius / (self.terrain_param * self.wheel_distance)]])
        self.B = np.array([[self.mass * self.wheel_radius2 / 4 + self.wheel_radius2 * self.Jzz / (
                self.terrain_param * self.wheel_distance2),
                            self.mass * self.wheel_radius2 / 4 - self.wheel_radius2 * self.Jzz / (
                                    self.terrain_param * self.wheel_distance2)],
                           [self.mass * self.wheel_radius2 / 4 - self.wheel_radius2 * self.Jzz / (
                                   self.terrain_param * self.wheel_distance2),
                            self.mass * self.wheel_radius2 / 4 + self.wheel_radius2 * self.Jzz / (
                                    self.terrain_param * self.wheel_distance2)]])
        # self.g_matrix = self.A @ np.linalg.inv(self.B)

        self.g_matrix = np.array(
            [[1 / self.wheel_radius, 1 / self.wheel_radius], [0, 0], [0, 0], [0, 0], [0, 0],
             [-self.vehicle_width / (2 * self.wheel_radius),
              self.vehicle_width / (2 * self.wheel_radius)]])
        self.g_matrix_dagger = np.matmul(np.linalg.inv(np.matmul(self.g_matrix.T, self.g_matrix)), self.g_matrix.T)

        self.maxForce = 10.0
        self.last_rpe = None
        self.last_Tkr1 = 0
        self.last_TRd = 0
        self.last_Tkr2 = 0
        self.last_energy = 0
        self.e1 = np.array([1, 0, 0]).reshape(3, 1)
        self.e2 = np.array([0, 1, 0]).reshape(3, 1)
        self.e3 = np.array([0, 0, 1]).reshape(3, 1)
        self.J = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        self.kp = 1.2
        self.kv = self.kp
        self.kr1 = 7
        self.kr2 = 3
        self.kw = 1

        self.Kp = self.kp * np.diag([1, 1, 1])
        self.Kv = self.kv * np.diag([1, 1, 1])
        self.KR1 = self.kr1 * np.diag([1, 1, 1])
        self.KR2 = self.kr2 * np.diag([1, 1, 1])
        self.Kw = self.kw * np.diag([1, 1, 1])

    def get_model(self, checkpoint=''):

        model = SE3HamNODE(device=device, udim=2,
                           pretrain=False, M1_known=self.M1_known,
                           M2_known=self.M2_known
                           ).to(device)
        # saved = "trained model unknown M1 M2"
        saved = ""
        using_pointcloud = True
        checkpoint = checkpoint
        if using_pointcloud:
            path = f'{THIS_DIR}/PointCloudTrainedModels/jackal-se3ham_pointclouds-rk4-5p' + checkpoint + '.tar'
        else:
            path = f'{THIS_DIR}/{saved}jackal-se3ham-rk4-5p' + checkpoint + '.tar'
        print(path)
        model.load_state_dict(torch.load(path, map_location=device))

        return model

    def Rpe_mat(self, v):
        '''
        Input (v)   : (2,1)-shaped numpy array
        Output (Rpe): (2,2)-shaped numpy array
        '''
        v_norm = np.linalg.norm(np.squeeze(v))
        Rpe = 1 / v_norm * np.column_stack((-v, -self.J @ v, v_norm * self.e3))
        return Rpe

    def hat_map(self, w):
        w_hat = np.array([[0, -w[2], w[1]],
                          [w[2], 0, -w[0]],
                          [-w[1], w[0], 0]])
        return w_hat

    def vee_map(self, R):
        arr_out = np.zeros(3)
        arr_out[0] = -R[1, 2]
        arr_out[1] = R[0, 2]
        arr_out[2] = -R[0, 1]
        return arr_out



    def get_control_new(self, currentState, targetState):
        """
        currentState: (18,)-shaped array containing current position, orientation, velocity, angular velocity.
        targetState: (18,)-shaped array containing target position,
                            target orientation, target velocity, target angular velocity
        """
        done = False
        pos, Rvec, vel, angvel = np.split(np.expand_dims(currentState, axis=1),
                                          [self.posdim, self.posdim + self.Rdim,
                                           self.posdim + self.Rdim + self.linveldim],
                                          axis=0)
        R = Rvec.reshape((3, 3))

        pos_des, Rvec_des, vel_des, angvel_des = np.split(np.expand_dims(targetState, axis=1),
                                                          [self.posdim, self.posdim + self.Rdim,
                                                           self.posdim + self.Rdim + self.linveldim], axis=0)
        R_des = Rvec_des.reshape((3, 3))
        acc_des = np.array([0, 0, 0]).reshape((3, 1))
        angacc_des = np.array([0, 0, 0]).reshape((3, 1))
        q = np.concatenate((pos[:3].reshape(-1, ), R.flatten()))
        q = torch.tensor(q, requires_grad=True, dtype=torch.float32).to(device)
        q = q.view(1, 12)
        xtensor, Rtensor = torch.split(q, [3, 9], dim=1)
        v = torch.tensor(vel, requires_grad=True, dtype=torch.float32).to(device).view(1, 3)
        omega = torch.tensor(angvel, requires_grad=True, dtype=torch.float32).to(device).view(1, 3)
        qp = torch.cat((q, v, omega), dim=1)
        g_q = self.gnet(q).detach().cpu().numpy()[0]
        g_q[1:5] = 0
        print(f"g matrix : {g_q}")
        g_matrix_dagger = np.linalg.inv(g_q.T @ g_q) @ g_q.T
        # g_matrix_dagger[1,0] = g_matrix_dagger[0, 0]
        # g_matrix_dagger[1, -1] = -g_matrix_dagger[0, -1]
        print(f"g dagger : {g_matrix_dagger}")
        pos_e = pos - pos_des
        Re = R_des.T @ R

        qe = np.concatenate((pos_e[:3].reshape(-1, ), Re.flatten()))
        qe = torch.tensor(qe, requires_grad=True, dtype=torch.float32).to(device)
        qe = qe.view(1, 12)
        xetensor, Retensor = torch.split(qe, [3, 9], dim=1)

        if not self.M1_known:
            self.M = np.linalg.inv(self.Mnet(xtensor).detach().cpu().numpy()[0])
        if not self.M2_known:
            self.Inertia = np.linalg.inv(self.Inet(Rtensor).detach().cpu().numpy()[0])
        D = self.Dnet(qp).detach().cpu().numpy()[0]
        Dv = D[:3, :3]
        Dw = D[3:, 3:]
        print(f"mass : {self.M} \nInertial: {self.Inertia}")

        # Current State
        pv = self.M @ vel
        pw = self.Inertia @ angvel

        yaw_error = mat2euler(Re)[-1]
        yaw_error = yaw_error
        pos_e_norm = np.linalg.norm(np.squeeze(pos_e))
        print("position error : {:.3f} || yaw error : {:.3f}".format(pos_e_norm, yaw_error))
        if pos_e_norm < 0.1 and np.abs(yaw_error) < 0.1:
            done = True
            self.compare_with_ground_truth_g_matrix(learnt_g=g_q)
            print(f"Reached close to goal")
            print(f"Distance error : {pos_e_norm} || yaw error : {yaw_error}")
            return np.array([0, 0]), done
        pos_e_norm = np.linalg.norm(np.squeeze(pos_e))

        Rpe = self.Rpe_mat(pos_e)

        R_perp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        theta2 = self.vee_map(logm((R_perp.T @ R_des.T @ Rpe)))
        Rd2 = R_perp @ expm(self.hat_map(theta2) / np.linalg.norm(theta2) * np.pi / 2)

        dRpedpe1 = (-pos_e[0, 0] / pos_e_norm ** 2) * Rpe - 1 / pos_e_norm * np.column_stack(
            (self.e1, self.J @ self.e1, np.zeros((3, 1))))
        dRpedpe2 = (-pos_e[1, 0] / pos_e_norm ** 2) * Rpe - 1 / pos_e_norm * np.column_stack(
            (self.e2, self.J @ self.e2, np.zeros((3, 1))))

        dRpedpe1 = np.array([[dRpedpe1[0, 0], dRpedpe1[0, 1], 0], [dRpedpe1[1, 0], dRpedpe1[1, 1], 0], [0, 0, 0]])
        dRpedpe2 = np.array([[dRpedpe2[0, 0], dRpedpe2[0, 1], 0], [dRpedpe2[1, 0], dRpedpe2[1, 1], 0], [0, 0, 0]])

        theta = self.vee_map(logm(R_des.T @ Rd2.T @ Rpe))
        Rd = expm(self.hat_map(theta) / np.linalg.norm(theta) * np.pi / 2)
        # print(mat_to_deg(Rd))
        re1_hat = self.hat_map(Re[0, :])
        re2_hat = self.hat_map(Re[1, :])
        re3_hat = self.hat_map(Re[2, :])

        eW = angvel - Re.T @ angvel_des
        ev = vel - R.T @ vel_des

        TkR1 = .5 * np.trace(np.eye(3) - R_des.T @ Rd2.T @ Rpe)
        TRd = .5 * np.trace(np.eye(3) - Rd.T @ Rpe.T @ Rd2 @ R_des @ Re)
        TkR2 = .5 * np.trace(np.eye(3) - Rpe.T @ Rd2 @ R_des @ Re)

        kr1trace = np.array(
            [np.trace(R_des.T @ Rd2.T @ dRpedpe1),
             np.trace(R_des.T @ Rd2.T @ dRpedpe2),
             0]).reshape((3, 1))

        krdtrace = np.array(
            [np.trace(Rd @ Re.T @ R_des.T @ Rd2.T @ dRpedpe1),
             np.trace(Rd @ Re.T @ R_des.T @ Rd2.T @ dRpedpe2),
             0]).reshape((3, 1))

        kr2trace = np.array(
            [np.trace(Re.T @ R_des.T @ Rd2.T @ dRpedpe1),
             np.trace(Re.T @ R_des.T @ Rd2.T @ dRpedpe2),
             0]).reshape((3, 1))

        dhdpe = self.kp * pos_e \
                - .5 * TRd * self.kr1 * kr1trace \
                - .5 * TkR1 * self.kr1 * krdtrace \
                - .5 * self.kr2 * kr2trace

        dHdRe = -.5 * self.kr1 * TkR1 * R_des.T @ Rd2.T @ Rpe @ Rd - .5 * self.kr2 * (R_des.T @ Rd2.T @ Rpe)

        # e_euler = 0.25 * self.vee_map(
        #     TkR1 * (Rd.T @ Rpe.T @ Rd2 @ R_des @ Re) - (Re.T @ R_des.T @ Rd2.T @ Rpe @ Rd) * TkR1
        # ).reshape(3, 1) + 0.5 * self.vee_map(
        #     self.kr2 * (Rpe.T @ Rd2 @ R_des @ Re) - (Re.T @ R_des.T @ Rd2.T @ Rpe) * self.kr2
        # ).reshape(3, 1)

        e_euler = re1_hat.T @ dHdRe[0, :].reshape(3, 1) \
                  + re2_hat.T @ dHdRe[1, :].reshape(3, 1) \
                  + re3_hat.T @ dHdRe[2, :].reshape(3, 1)

        bw = -e_euler - self.Kw @ eW \
             - self.hat_map(np.squeeze(pv)) @ vel \
             - self.hat_map(np.squeeze(pw)) @ angvel - Dw @ angvel

        bv = -R.T @ dhdpe - self.Kv @ ev - self.hat_map(np.squeeze(pv)) @ angvel \
             + self.M @ (R.T @ acc_des - self.hat_map(np.squeeze(angvel)) @ R.T @ vel_des) - Dv @ vel

        wrench = np.hstack((np.squeeze(bv), np.squeeze(bw)))
        u = g_matrix_dagger @ wrench
        u1 = u[0]
        u2 = u[1]

        tau_L = u1
        tau_R = u2
        u = np.array([tau_L, tau_R])
        u = np.clip(u, -self.maxTorque, self.maxTorque)
        energy = 1/2 * (pv.T @ np.linalg.inv(self.M) @ pv + pw.T @ np.linalg.inv(self.Inertia) @ pw)
        print(energy)
        return u, done

    def compare_with_ground_truth_g_matrix(self, learnt_g):
        print(f"learnt g : {learnt_g}")
        print(f"ground truth g : {self.g_matrix}")
