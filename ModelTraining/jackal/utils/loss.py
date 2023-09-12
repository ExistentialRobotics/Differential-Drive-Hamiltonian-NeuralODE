# This is modified from: https://github.com/thaipduong/SE3HamDL
import torch


def L2_loss(u, v):
    return (u - v).pow(2).mean()


def point_cloud_l2_loss_for_successive_point_clouds(
        observedPointClouds,
        predictedStates,
        split
):
    """
    1. observedPointClouds is the observed point clouds from robot of shape 2 X 5 X P
        First dimension is source point cloud points and target point clouds
        Second dimension is x,y,z, tauR, tauL associated with the point cloud where tauR, tauL is the control input
        Third dimension is the number of points per point cloud
    2. predictedStates is the robot state predicition from the torch ode at each of the sample time stamps
    3. split is the data split indices for x, R, v, w, tauR, tauL from concatenated tensor
    """
    num_particles = observedPointClouds.shape[-1]
    samples_per_control = observedPointClouds.shape[0]
    # print(f"observation shapes : {observedPointClouds.shape}")
    # print(f"predicted state shapes : {predictedStates.shape}")
    # 4X480X3X20 --> 4X480X20X3 --> 4 X 480 X 2 X 20 X 3
    x = observedPointClouds.permute(0, 1, 2, 4, 3)  # 4 X 480 X 2 X 5 X 20 --> 4 X 480 X 2 X 20 X 5
    x_hat, R_hat, q_dot_hat, u_hat = torch.split(predictedStates, split, dim=2)
    R_hat = R_hat.flatten(start_dim=0, end_dim=1)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R_hat = norm_R_hat.reshape(predictedStates.shape[0], predictedStates.shape[1], 3, 3)

    R_hat = norm_R_hat
    R_hat_transpose = R_hat.transpose(-2, -1)

    R_hat_cat = R_hat.repeat(num_particles, 1, 1, 1, 1)
    R_hat_cat = R_hat_cat.permute(1, 2, 0, 3, 4)
    R_hat_transpose_cat = R_hat_transpose.repeat(num_particles, 1, 1, 1, 1)
    R_hat_transpose_cat = R_hat_transpose_cat.permute(1, 2, 0, 3, 4)
    x_hat_cat = x_hat.repeat(num_particles, 1, 1, 1)
    x_hat_cat = x_hat_cat.permute(1, 2, 0, 3)
    error_cat = None
    for i in range(0, samples_per_control, 3):  # p_R2 =  R2_T_w @ p_w = w_T_R2 ^-1 @ p_w = [w_R_2  w_t_2; 0 , 1] ^-1 @ p_w
        source_pc = x[i, :, 0, :, :3]
        target_pc = x[i, :, 1, :, :3]

        z1w = R_hat_cat[i, :, :, :, :] @ torch.unsqueeze(source_pc, dim=3) + torch.unsqueeze(x_hat_cat[i, :, :, :],
                                                                                             dim=3)
        z2b = R_hat_transpose_cat[i + 1, :, :, :, :] @ z1w - R_hat_transpose_cat[i + 1, :, :, :, :] \
              @ torch.unsqueeze(x_hat_cat[i + 1, :, :, :], dim=3)
        error = torch.squeeze(torch.unsqueeze(target_pc, dim=3) - z2b)
        if i == 0:
            error_cat = torch.unsqueeze(error, dim=0)
        else:
            error_cat = torch.cat((error_cat, torch.unsqueeze(error, dim=0)), dim=0)
    # print(f"error: {error_cat}")
    error_final = error_cat.flatten(start_dim=0, end_dim=2)
    cost = torch.norm(error_final, dim=1).mean()
    return cost


def point_cloud_L2_loss(c, c_hat, split):
    # z2 - ((R1' * (p2-p1)) - z1)
    num_particles = c.shape[3]
    samples_per_control = c_hat.shape[0]

    x = c.permute(0, 1, 3, 2)
    x_hat, R_hat, q_dot_hat, u_hat = torch.split(c_hat, split, dim=2)
    R_hat = R_hat.flatten(start_dim=0, end_dim=1)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R_hat = norm_R_hat.reshape(c_hat.shape[0], c_hat.shape[1], 3, 3)
    # norm_R_hat = norm_R_hat.reshape(c_hat.shape[1], c_hat.shape[0], 3, 3)
    # norm_R_hat = norm_R_hat.permute(1,0,2,3)

    # R_hat = R_hat.reshape(c_hat.shape[0], c_hat.shape[1], 3, 3)
    R_hat = norm_R_hat
    R_hat_transpose = R_hat.transpose(-2, -1)

    R_hat_cat = R_hat.repeat(num_particles, 1, 1, 1, 1)
    R_hat_cat = R_hat_cat.permute(1, 2, 0, 3, 4)
    R_hat_transpose_cat = R_hat_transpose.repeat(num_particles, 1, 1, 1, 1)
    R_hat_transpose_cat = R_hat_transpose_cat.permute(1, 2, 0, 3, 4)
    x_hat_cat = x_hat.repeat(num_particles, 1, 1, 1)
    x_hat_cat = x_hat_cat.permute(1, 2, 0, 3)

    error_cat = None
    for i in range(samples_per_control - 1):
        z1w = R_hat_cat[i, :, :, :, :] @ torch.unsqueeze(x[i, :, :, :], dim=3) + torch.unsqueeze(x_hat_cat[i, :, :, :],
                                                                                                 dim=3)
        z2b = R_hat_transpose_cat[i + 1, :, :, :, :] @ z1w - R_hat_transpose_cat[i + 1, :, :, :, :] @ torch.unsqueeze(
            x_hat_cat[i + 1, :, :, :], dim=3)
        error = torch.squeeze(torch.unsqueeze(x[i + 1, :, :, :], dim=3) - z2b)
        if (i == 0):
            error_cat = torch.unsqueeze(error, dim=0)
        else:
            error_cat = torch.cat((error_cat, torch.unsqueeze(error, dim=0)), dim=0)
    error_final = error_cat.flatten(start_dim=0, end_dim=2)
    cost = torch.norm(error_final, dim=1).mean()
    return cost


def pose_L2_geodesic_loss(u, u_hat, split):
    #################
    x_hat, R_hat, q_dot_hat, u_hat = torch.split(u_hat, split, dim=2)
    x, R, q_dot, u = torch.split(u, split, dim=2)
    v_hat, w_hat = torch.split(q_dot_hat, [3, 3], dim=2)
    v, w = torch.split(q_dot, [3, 3], dim=2)

    v = v.flatten(start_dim=0, end_dim=1)
    v_hat = v_hat.flatten(start_dim=0, end_dim=1)
    vloss = L2_loss(v, v_hat)
    # print("vloss: ", vloss.detach().cpu().numpy())
    w = w.flatten(start_dim=0, end_dim=1)
    w_hat = w_hat.flatten(start_dim=0, end_dim=1)
    wloss = L2_loss(w, w_hat)
    # print("wloss: ", wloss.detach().cpu().numpy())
    # wmean = w_hat.pow(2).mean()
    # print("wmean: ", wmean.detach().cpu().numpy())
    # x_qdot_u_hat = torch.cat((x_hat, q_dot_hat, u_hat), dim=1)
    # x_qdot_u = torch.cat((x, q_dot, u), dim=1)
    x = x.flatten(start_dim=0, end_dim=1)
    x_hat = x_hat.flatten(start_dim=0, end_dim=1)
    x_loss = L2_loss(x, x_hat)
    R = R.flatten(start_dim=0, end_dim=1)
    R_hat = R_hat.flatten(start_dim=0, end_dim=1)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R = compute_rotation_matrix_from_unnormalized_rotmat(R)
    geo_loss, _ = compute_geodesic_loss(norm_R, norm_R_hat)
    return x_loss + vloss + wloss + geo_loss, x_loss, vloss, wloss, geo_loss


def pose_L2_geodesic_diff(u, u_hat, split):
    #################
    x_hat, R_hat, q_dot_hat, u_hat = torch.split(u_hat, split, dim=1)
    x, R, q_dot, u = torch.split(u, split, dim=1)
    x_qdot_u_hat = torch.cat((x_hat, q_dot_hat, u_hat), dim=1)
    x_qdot_u = torch.cat((x, q_dot, u), dim=1)
    l2_diff = torch.sum((x_qdot_u - x_qdot_u_hat) ** 2, dim=1)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R = compute_rotation_matrix_from_unnormalized_rotmat(R)
    _, geo_diff = compute_geodesic_loss(norm_R, norm_R_hat)
    return l2_diff + geo_diff, l2_diff, geo_diff


def traj_pose_L2_geodesic_loss(traj, traj_hat, split):
    total_loss = None
    l2_loss = None
    geo_loss = None
    for t in range(traj.shape[0]):
        u = traj[t, :, :]
        u_hat = traj_hat[t, :, :]
        if total_loss is None:
            total_loss, l2_loss, geo_loss = pose_L2_geodesic_diff(u, u_hat, split=split)
            total_loss = torch.unsqueeze(total_loss, dim=0)
            l2_loss = torch.unsqueeze(l2_loss, dim=0)
            geo_loss = torch.unsqueeze(geo_loss, dim=0)
        else:
            t_total_loss, t_l2_loss, t_geo_loss = pose_L2_geodesic_diff(u, u_hat, split=split)
            t_total_loss = torch.unsqueeze(t_total_loss, dim=0)
            t_l2_loss = torch.unsqueeze(t_l2_loss, dim=0)
            t_geo_loss = torch.unsqueeze(t_geo_loss, dim=0)
            total_loss = torch.cat((total_loss, t_total_loss), dim=0)
            l2_loss = torch.cat((l2_loss, t_l2_loss), dim=0)
            geo_loss = torch.cat((geo_loss, t_geo_loss), dim=0)
    return total_loss, l2_loss, geo_loss


def compute_geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    theta = theta ** 2
    error = theta.mean()
    return error, theta


def compute_rotation_matrix_from_quaternion(quaternion):
    batch = quaternion.shape[0]

    quat = torch.nn.functional.normalize(quaternion)  # normalize_vector(quaternion).contiguous()

    qw = quat[..., 0].contiguous().view(batch, 1)
    qx = quat[..., 1].contiguous().view(batch, 1)
    qy = quat[..., 2].contiguous().view(batch, 1)
    qz = quat[..., 3].contiguous().view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    if v.is_cuda:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
    return out


def compute_rotation_matrix_from_unnormalized_rotmat(unnormalized_rotmat):
    x_raw = unnormalized_rotmat[:, 0:3]  # batch*3
    y_raw = unnormalized_rotmat[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 1, 3)
    y = y.view(-1, 1, 3)
    z = z.view(-1, 1, 3)
    matrix = torch.cat((x, y, z), 1)  # batch*3*3
    return matrix


# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    if m1.is_cuda:
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
    else:
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch)))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch)) * -1)

    theta = torch.acos(cos)
    return theta
