import torch
import numpy as np
from se3hamneuralode import MLP, PSDMass, PSDInertial, MatrixNet
from se3hamneuralode import compute_rotation_matrix_from_quaternion
from .utils import L2_loss


class SE3HamNODE(torch.nn.Module):
    """
    class representing the SE3 Hamiltonian Neural ODE network architecture

    """
    def __init__(self, device=None, pretrain=True, M_net1=None, M_net2=None,
                 V_net=None, g_net=None, udim=2, M1_known: bool = False, M2_known: bool = False):
        super(SE3HamNODE, self).__init__()
        init_gain = 0.001
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim  # 3 for position + 12 for rotmat
        self.twistdim = self.linveldim + self.angveldim  # 3 for linear vel + 3 for ang vel
        self.udim = udim
        self.M1_known = M1_known
        self.M2_known = M2_known
        if M_net1 is None and self.M1_known is False:
            print("Creating M1 net")
            self.M_net1 = PSDMass(self.xdim, 20, self.linveldim, init_gain=init_gain).to(device)
        else:
            self.M_net1 = M_net1
        if M_net2 is None and self.M2_known is False:
            print("Creating M2 net")
            self.M_net2 = PSDInertial(self.Rdim, 10, self.twistdim - self.linveldim, init_gain=init_gain).to(device)
        else:
            self.M_net2 = M_net2
        if V_net is None:
            self.V_net = MLP(self.posedim, 2, 1, init_gain=init_gain).to(device)
        else:
            self.V_net = V_net
        if g_net is None:
            self.g_net = MatrixNet(self.posedim, 30, self.twistdim * self.udim, shape=(self.twistdim, self.udim),
                                   init_gain=init_gain).to(device)
        else:
            self.g_net = g_net
        self.D_net = PSDInertial(
            input_dim=self.posedim + self.linveldim + self.angveldim, hidden_dim=80,
            diag_dim=self.linveldim + self.angveldim, init_gain=init_gain
        ).to(device)
        self.device = device
        self.nfe = 0
        if pretrain:
            self.pretrain()

    def pretrain(self):
        """
        pretrain the M1, M2 and g networks to incorporate prior information / fit to nominal values
        """
        if not self.M1_known:
            x = np.arange(-10, 10, 0.25)
            y = np.arange(-10, 10, 0.25)
            z = np.arange(-10, 10, 0.25)
            n_grid = len(z)
            batch = n_grid ** 3
            xx, yy, zz = np.meshgrid(x, y, z)
            Xgrid = np.zeros([batch, 3])
            Xgrid[:, 0] = np.reshape(xx, (batch,))
            Xgrid[:, 1] = np.reshape(yy, (batch,))
            Xgrid[:, 2] = np.reshape(zz, (batch,))
            Xgrid = torch.tensor(Xgrid, dtype=torch.float32).view(batch, 3).to(self.device)
            # Pretain M_net1
            m_net1_hat = self.M_net1(Xgrid)
            # Train M_net1 to output identity matrix
            m_guess = torch.eye(3)
            m_guess = m_guess.reshape((1, 3, 3))
            m_guess = m_guess.repeat(batch, 1, 1).to(self.device)
            optim1 = torch.optim.Adam(self.M_net1.parameters(), 1e-3, weight_decay=0.0)
            loss = L2_loss(m_net1_hat, m_guess)
            print("Start pretraining Mnet1!", loss.detach().cpu().numpy())
            step = 1
            while loss > 1e-6:
                loss.backward()
                optim1.step()
                optim1.zero_grad()
                if step % 10 == 0:
                    print("step", step, loss.detach().cpu().numpy())
                m_net1_hat = self.M_net1(Xgrid)
                loss = L2_loss(m_net1_hat, m_guess)
                step = step + 1
            print("Pretraining Mnet1 done!", loss.detach().cpu().numpy())
            # delete Xgrid to save memory
            del Xgrid
            torch.cuda.empty_cache()

        if not self.M2_known:
            # Pretrain M_net2
            batch = 250000
            # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
            rand_ = np.random.uniform(size=(batch, 3))
            u1, u2, u3 = rand_[:, 0], rand_[:, 1], rand_[:, 2]
            quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                             np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
            q_tensor = torch.tensor(quat.transpose(), dtype=torch.float32).view(batch, 4).to(self.device)
            R_tensor = compute_rotation_matrix_from_quaternion(q_tensor)
            R_tensor = R_tensor.view(-1, 9)
            m_net2_hat = self.M_net2(R_tensor)
            # Train M_net2 to output identity matrix
            inertia_guess = torch.eye(3)
            inertia_guess = inertia_guess.reshape((1, 3, 3))
            inertia_guess = inertia_guess.repeat(batch, 1, 1).to(self.device)
            optim = torch.optim.Adam(self.M_net2.parameters(), 1e-3, weight_decay=0.0)
            loss = L2_loss(m_net2_hat, inertia_guess)
            print("Start pretraining Mnet2!", loss.detach().cpu().numpy())
            step = 1
            while loss > 1e-6:
                loss.backward()
                optim.step()
                optim.zero_grad()
                if step % 10 == 0:
                    print("step", step, loss.detach().cpu().numpy())
                m_net2_hat = self.M_net2(R_tensor)
                loss = L2_loss(m_net2_hat, inertia_guess)
                step = step + 1
            print("Pretraining Mnet2 done!", loss.detach().cpu().numpy())
            # Delete data and cache to save memory
            del q_tensor
            torch.cuda.empty_cache()

        # pretrain g_q to have desired structure which enables us to learn only residual
        x = np.arange(-1, 1.05, 0.05)
        y = np.arange(-1, 1.05, 0.05)
        z = np.arange(-1, 1.05, 0.05)
        n_grid = len(z)
        batch = n_grid ** 3
        xx, yy, zz = np.meshgrid(x, y, z)
        Xgrid = np.zeros([batch, 3])
        Xgrid[:, 0] = np.reshape(xx, (batch,))
        Xgrid[:, 1] = np.reshape(yy, (batch,))
        Xgrid[:, 2] = np.reshape(zz, (batch,))
        Xgrid = torch.tensor(Xgrid, dtype=torch.float32).view(batch, 3).to(self.device)
        # Uniformly generate quaternion using http://planning.cs.uiuc.edu/node198.html
        rand_ = np.random.uniform(size=(batch, 3))
        u1, u2, u3 = rand_[:, 0], rand_[:, 1], rand_[:, 2]
        quat = np.array([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                         np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)])
        q_tensor = torch.tensor(quat.transpose(), dtype=torch.float32).view(batch, 4).to(self.device)
        R_tensor = compute_rotation_matrix_from_quaternion(q_tensor)
        R_tensor = R_tensor.view(-1, 9)

        q_sampled = torch.cat([Xgrid, R_tensor], dim=1)
        g_q_hat = self.g_net(q_sampled)
        g_q_structure = torch.from_numpy(
            np.array(
                [
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [-1, 1]
                ]
            )
        ).float()
        g_q_structure = g_q_structure.reshape((1, 6, 2))
        g_q_structure = g_q_structure.repeat((batch, 1, 1)).to(self.device)
        optim = torch.optim.Adam(self.g_net.parameters(), 1e-3, weight_decay=0.0)
        loss = L2_loss(g_q_hat, g_q_structure)
        print("Start pretraining gnet!", loss.detach().cpu().numpy())
        step = 1
        while loss > 1e-6:
            loss.backward()
            optim.step()
            optim.zero_grad()
            if step % 10 == 0:
                print("step", step, loss.detach().cpu().numpy())
            g_q_hat = self.g_net(q_sampled)
            loss = L2_loss(g_q_hat, g_q_structure)
            step = step + 1
        print("Pretraining gnet done!", loss.detach().cpu().numpy())
        # Delete data and cache to save memory
        del q_tensor, R_tensor, q_sampled
        torch.cuda.empty_cache()

    def forward(self, t, input):
        with torch.enable_grad():
            mass = 9
            Ixx, Iyy, Izz = 0.25, 0.25, 0.33
            self.nfe += 1
            q, q_dot, u = torch.split(input, [self.posedim, self.twistdim, self.udim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)
            q_dot_v, q_dot_w = torch.split(q_dot, [self.linveldim, self.angveldim], dim=1)

            if self.M1_known is False:
                M_q_inv1 = self.M_net1(x)
            else:
                M_q_inv1 = torch.eye(3) * 1 / mass
                M_q_inv1 = M_q_inv1.reshape(1, 3, 3)
                M_q_inv1 = M_q_inv1.repeat(q.shape[0], 1, 1).to(self.device)

            if self.M2_known is False:
                M_q_inv2 = self.M_net2(R)
            else:
                M_q_inv2 = torch.tensor(
                    np.diag([1/Ixx, 1/Iyy, 1/Izz]),
                    dtype=torch.float32
                )
                M_q_inv2 = M_q_inv2.reshape(1, 3, 3)
                M_q_inv2 = M_q_inv2.repeat(q.shape[0], 1, 1).to(self.device)

            q_dot_aug_v = torch.unsqueeze(q_dot_v, dim=2)
            q_dot_aug_w = torch.unsqueeze(q_dot_w, dim=2)
            pv = torch.squeeze(torch.matmul(torch.inverse(M_q_inv1), q_dot_aug_v), dim=2)
            pw = torch.squeeze(torch.matmul(torch.inverse(M_q_inv2), q_dot_aug_w), dim=2)
            q_p = torch.cat((q, pv, pw), dim=1)
            q, pv, pw = torch.split(q_p, [self.posedim, self.linveldim, self.angveldim], dim=1)
            x, R = torch.split(q, [self.xdim, self.Rdim], dim=1)

            # Neural networks' forward passes
            if self.M1_known is False:
                M_q_inv1 = self.M_net1(x)
            else:
                M_q_inv1 = torch.eye(3) * 1 / mass
                M_q_inv1 = M_q_inv1.reshape(1, 3, 3)
                M_q_inv1 = M_q_inv1.repeat(q.shape[0], 1, 1).to(self.device)

            if self.M2_known is False:
                M_q_inv2 = self.M_net2(R)
            else:
                M_q_inv2 = torch.tensor(
                    np.diag([1 / Ixx, 1 / Iyy, 1 / Izz]),
                    dtype=torch.float32
                )
                M_q_inv2 = M_q_inv2.reshape(1, 3, 3)
                M_q_inv2 = M_q_inv2.repeat(q.shape[0], 1, 1).to(self.device)
            V_q = 0
            g_q = self.g_net(q)
            D_qp = self.D_net(q_p)

            # Calculate the Hamiltonian
            p_aug_v = torch.unsqueeze(pv, dim=2)
            p_aug_w = torch.unsqueeze(pw, dim=2)
            H = torch.squeeze(torch.matmul(torch.transpose(p_aug_v, 1, 2), torch.matmul(M_q_inv1, p_aug_v))) / 2.0 + \
                torch.squeeze(torch.matmul(torch.transpose(p_aug_w, 1, 2), torch.matmul(M_q_inv2, p_aug_w))) / 2.0

            if self.M1_known is False and self.M2_known is False:
                # Calculate the partial derivative using autograd
                dH = torch.autograd.grad(H.sum(), q_p, create_graph=True)[0]
                # Order: position (3), rotmat (9), lin vel (3) in body frame, ang vel (3) in body frame
                dHdx, dHdR, dHdpv, dHdpw = torch.split(dH, [self.xdim, self.Rdim, self.linveldim, self.angveldim], dim=1)
            else:
                # if M1 and M2 are known, the derivatives wrt x, R are simply zeros,
                # Derivative of H wrt pv and pw are simply M^{-1} @ p
                dHdpv = torch.squeeze(M_q_inv1 @ p_aug_v, dim=2)
                dHdpw = torch.squeeze(M_q_inv2 @ p_aug_w, dim=2)
                dHdx = torch.zeros(dHdpv.shape).to(self.device)
                dHdR = torch.zeros((dHdpv.shape[0], 9)).to(self.device)

            # Calculate g*u
            F = torch.squeeze(torch.matmul(g_q, torch.unsqueeze(u, dim=2)))

            # Hamilton's equation on SE(3) manifold for (q,p)
            Rmat = R.view(-1, 3, 3)
            dx = torch.squeeze(torch.matmul(Rmat, torch.unsqueeze(dHdpv, dim=2)))
            dR03 = torch.cross(Rmat[:, 0, :], dHdpw)
            dR36 = torch.cross(Rmat[:, 1, :], dHdpw)
            dR69 = torch.cross(Rmat[:, 2, :], dHdpw)
            dR = torch.cat((dR03, dR36, dR69), dim=1)

            dpv = torch.cross(pv, dHdpw) \
                  - torch.squeeze(torch.matmul(torch.transpose(Rmat, 1, 2), torch.unsqueeze(dHdx, dim=2))) \
                  + F[:, 0:3] - torch.squeeze(torch.matmul(D_qp[:, :3, :3], torch.unsqueeze(dHdpv, dim=2)))

            dpw = torch.cross(pw, dHdpw) \
                  + torch.cross(pv, dHdpv) \
                  + torch.cross(Rmat[:, 0, :], dHdR[:, 0:3]) \
                  + torch.cross(Rmat[:, 1, :], dHdR[:, 3:6]) \
                  + torch.cross(Rmat[:, 2, :], dHdR[:, 6:9]) \
                  + F[:, 3:6] - torch.squeeze(torch.matmul(D_qp[:, 3:, 3:], torch.unsqueeze(dHdpw, dim=2)))
            # Hamilton's equation on SE(3) manifold for twist xi

            dM_inv_dt1 = torch.zeros_like(M_q_inv1)
            if self.M1_known is False:
                for row_ind in range(self.linveldim):
                    for col_ind in range(self.linveldim):
                        dM_inv1 = \
                            torch.autograd.grad(M_q_inv1[:, row_ind, col_ind].sum(), x, create_graph=True, allow_unused = True)[0]
                        dM_inv_dt1[:, row_ind, col_ind] = (dM_inv1 * dx).sum(-1)

            dv = torch.squeeze(torch.matmul(M_q_inv1, torch.unsqueeze(dpv, dim=2)), dim=2) \
                 + torch.squeeze(torch.matmul(dM_inv_dt1, torch.unsqueeze(pv, dim=2)), dim=2)

            dM_inv_dt2 = torch.zeros_like(M_q_inv2)

            if self.M2_known is False:
                for row_ind in range(self.angveldim):
                    for col_ind in range(self.angveldim):
                        dM_inv2 = \
                            torch.autograd.grad(M_q_inv2[:, row_ind, col_ind].sum(), R, create_graph=True, allow_unused = True)[0]
                        dM_inv_dt2[:, row_ind, col_ind] = (dM_inv2 * dR).sum(-1)
            dw = torch.squeeze(torch.matmul(M_q_inv2, torch.unsqueeze(dpw, dim=2)), dim=2) \
                 + torch.squeeze(torch.matmul(dM_inv_dt2, torch.unsqueeze(pw, dim=2)), dim=2)

            batch_size = input.shape[0]
            zero_vec = torch.zeros(batch_size, self.udim, dtype=torch.float32, device=self.device)
            return torch.cat((dx, dR, dv, dw, zero_vec), dim=1)
