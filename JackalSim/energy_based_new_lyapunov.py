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

    def __init__(self, args=None, maxTorque=2, Kp = None, Kv = None, KR1 = None, KR2 = None, Kw = None):
        # Kp = 10.0 * np.diag([5, 5, 25])
        # Kv = 10.0 * np.diag([2.5, 2.5, 2.5])
        # KR1 = 0.01 * np.diag([250, 250, 250])
        # KR2 = 0.01 * np.diag([250, 250, 250])
        # Kw = 3 * np.diag([2, 2, 2])

        # Dimensions
        self.posdim = 3
        self.eulerdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.qdim = self.posdim + self.eulerdim + self.linveldim + self.angveldim
        self.udim = 2

        # Parameters
        self.mass = 8.5
        self.Jxx = 0.25
        self.Jyy = 0.25
        self.Jzz = 0.33
        self.vehicle_width = 0.5
        self.wheel_radius = 0.1
        self.M = self.mass * np.eye(3)
        self.Inertia = np.diag([self.Jxx, self.Jyy, self.Jzz])

        self.wheel_radius2 = self.wheel_radius * self.wheel_radius
        self.wheel_distance = 0.5
        self.wheel_distance2 = self.wheel_distance * self.wheel_distance
        self.terrain_param = 1

        self.Jzz = 0.33

        self.g_matrix = np.array(
            [[1 / self.wheel_radius, 1 / self.wheel_radius], [0, 0], [0, 0], [0, 0], [0, 0],
             [-self.vehicle_width / (2 * self.wheel_radius),
              self.vehicle_width / (2 * self.wheel_radius)]])
        self.g_matrix_dagger = np.matmul(np.linalg.inv(np.matmul(self.g_matrix.T, self.g_matrix)), self.g_matrix.T)

        self.maxTorque = maxTorque

        self.last_rpe = None
        self.last_Tkr1 = 0
        self.last_TRd = 0
        self.last_Tkr2 = 0
        self.last_energy = 0
        self.e1 = np.array([1, 0, 0]).reshape(3, 1)
        self.e2 = np.array([0, 1, 0]).reshape(3, 1)
        self.e3 = np.array([0, 0, 1]).reshape(3, 1)
        self.J = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        # Energy based control parameters
        self.kp = 1.6
        self.kv = self.kp
        self.kr1 = 15
        self.kr2 = 6
        self.kw = 1

        self.Kp = self.kp * np.diag([1, 1, 1])
        self.Kv = self.kv * np.diag([1, 1, 1])
        self.KR1 = self.kr1 * np.diag([1, 1, 1])
        self.KR2 = self.kr2 * np.diag([1, 1, 1])
        self.Kw = self.kw * np.diag([1, 1, 1])

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

    def get_control(self, currentState, targetState):
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

        # Current State 
        pv = self.M @ vel
        pw = self.Inertia @ angvel
        
        # Matrices
        pos_e = (pos - pos_des)
        Re = R_des.T @ R
        yaw_error = mat2euler(Re)[-1]
        yaw_error = yaw_error
        pos_e_norm = np.linalg.norm(np.squeeze(pos_e))
        if pos_e_norm < 0.005 and np.abs(yaw_error) < 0.05:
            print(f"Reached close to goal")
            print(f"Distance error : {pos_e_norm} || yaw error : {yaw_error}")
            return np.array([0, 0]), True

        Rpe = self.Rpe_mat(pos_e)
        R_perp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        theta2 = vee_map(logm((R_perp.T @ R_des.T @ Rpe)))
        Rd2 = R_perp @ expm(hat_map(theta2) / np.linalg.norm(theta2) * np.pi / 2)

        dRpedpe1 = (-pos_e[0, 0] / pos_e_norm ** 2) * Rpe - 1 / pos_e_norm * np.column_stack(
            (self.e1, self.J @ self.e1, np.zeros((3, 1))))
        dRpedpe2 = (-pos_e[1, 0] / pos_e_norm ** 2) * Rpe - 1 / pos_e_norm * np.column_stack(
            (self.e2, self.J @ self.e2, np.zeros((3, 1))))

        dRpedpe1 = np.array([[dRpedpe1[0, 0], dRpedpe1[0, 1], 0], [dRpedpe1[1, 0], dRpedpe1[1, 1], 0], [0, 0, 0]])
        dRpedpe2 = np.array([[dRpedpe2[0, 0], dRpedpe2[0, 1], 0], [dRpedpe2[1, 0], dRpedpe2[1, 1], 0], [0, 0, 0]])

        theta = vee_map(logm(R_des.T @ Rd2.T @ Rpe))
        Rd = expm(hat_map(theta) / np.linalg.norm(theta) * np.pi / 2)
        re1_hat = hat_map(Re[0, :])
        re2_hat = hat_map(Re[1, :])
        re3_hat = hat_map(Re[2, :])

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

        e_euler = re1_hat.T @ dHdRe[0, :].reshape(3, 1) \
                  + re2_hat.T @ dHdRe[1, :].reshape(3, 1) \
                  + re3_hat.T @ dHdRe[2, :].reshape(3, 1)

        bw = -e_euler - self.Kw @ eW \
             - hat_map(np.squeeze(pv)) @ vel \
             - hat_map(np.squeeze(pw)) @ angvel

        bv = -R.T @ dhdpe - self.Kv @ ev - hat_map(np.squeeze(pv)) @ angvel \
             + self.M @ (R.T @ acc_des - hat_map(np.squeeze(angvel)) @ R.T @ vel_des)
        wrench = np.hstack((np.squeeze(bv), np.squeeze(bw)))
        u = self.g_matrix_dagger @ wrench
        u1 = u[0]
        u2 = u[1]

        tau_L = u1
        tau_R = u2
        u = np.array([tau_L, tau_R])
        u = np.clip(u, -self.maxTorque, self.maxTorque)
        return u, done
