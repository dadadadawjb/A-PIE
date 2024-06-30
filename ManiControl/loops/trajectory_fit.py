from typing import Tuple
import numpy as np

def fit_arc(points:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # points: (N, 3), C: (3,), A: (3,)
    # from https://blog.csdn.net/jiangjjp2812/article/details/106937333
    N = points.shape[0]
    assert N >= 3

    # fit plane, ax+by+cz=1
    try:
        A = np.linalg.inv(points.T @ points) @ points.T @ np.ones((N, 1))
    except np.linalg.LinAlgError:
        A = np.linalg.inv(points.T @ points + 1e-5 * np.identity(3)) @ points.T @ np.ones((N, 1))
    
    # fit circle, C * (P1+P2)/2 = 0
    B = np.zeros((N*(N-1)//2, 3))
    L = np.zeros((N*(N-1)//2, 1))
    for i in range(N-1):
        for j in range(i+1, N):
            B[i*(2*N-1-i)//2+j-i-1, :] = points[j, :] - points[i, :]
            L[i*(2*N-1-i)//2+j-i-1, 0] = (np.linalg.norm(points[j, :])**2 - np.linalg.norm(points[i, :])**2) / 2
    coefficient_matrix = np.zeros((4, 4))
    coefficient_matrix[:3, :3] = B.T @ B
    coefficient_matrix[:3, 3:] = A
    coefficient_matrix[3:, :3] = A.T
    coefficient_matrix[3, 3] = 0
    rhs = np.zeros((4, 1))
    rhs[:3, 0:] = B.T @ L
    rhs[3, 0] = 1
    try:
        C_lambda = np.linalg.inv(coefficient_matrix) @ rhs
    except np.linalg.LinAlgError:
        C_lambda = np.linalg.inv(coefficient_matrix + 1e-5 * np.identity(4)) @ rhs
    C = C_lambda[:3, 0]

    return (C, (A / np.linalg.norm(A)).squeeze(axis=-1))


def fit_line(points:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # points: (N, 3), pivot: (3,), direction: (3,)
    # from https://blog.csdn.net/weixin_42203839/article/details/105327400
    N = points.shape[0]
    assert N >= 2

    x_limit = np.max(points[:, 0]) - np.min(points[:, 0])
    y_limit = np.max(points[:, 1]) - np.min(points[:, 1])
    z_limit = np.max(points[:, 2]) - np.min(points[:, 2])
    if max(x_limit, y_limit, z_limit) == x_limit:
        # fit line, y=ax+b z=cx+d
        coefficient_matrix = np.zeros((2, 2))
        coefficient_matrix[0, 0] = np.sum(points[:, 0]**2)
        coefficient_matrix[0, 1] = np.sum(points[:, 0])
        coefficient_matrix[1, 0] = coefficient_matrix[0, 1]
        coefficient_matrix[1, 1] = N
        rhs = np.zeros((2, 2))
        rhs[0, 0] = np.sum(points[:, 0] * points[:, 1])
        rhs[0, 1] = np.sum(points[:, 1])
        rhs[1, 0] = np.sum(points[:, 0] * points[:, 2])
        rhs[1, 1] = np.sum(points[:, 2])
        try:
            abcd = rhs @ np.linalg.inv(coefficient_matrix)
        except np.linalg.LinAlgError:
            abcd = rhs @ np.linalg.inv(coefficient_matrix + 1e-5 * np.identity(2))
        a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
        direction = np.array([1, a, c]) / np.linalg.norm(np.array([1, a, c]))
        x_positive = np.dot(direction, np.array([1, 0, 0])) > 0
        if x_positive:
            x_min = np.min(points[:, 0])
            pivot = np.array([x_min, a * x_min + b, c * x_min + d])
        else:
            x_max = np.max(points[:, 0])
            pivot = np.array([x_max, a * x_max + b, c * x_max + d])
    elif max(x_limit, y_limit, z_limit) == y_limit:
        # fit line, x=ay+b z=cy+d
        coefficient_matrix = np.zeros((2, 2))
        coefficient_matrix[0, 0] = np.sum(points[:, 1]**2)
        coefficient_matrix[0, 1] = np.sum(points[:, 1])
        coefficient_matrix[1, 0] = coefficient_matrix[0, 1]
        coefficient_matrix[1, 1] = N
        rhs = np.zeros((2, 2))
        rhs[0, 0] = np.sum(points[:, 0] * points[:, 1])
        rhs[0, 1] = np.sum(points[:, 0])
        rhs[1, 0] = np.sum(points[:, 1] * points[:, 2])
        rhs[1, 1] = np.sum(points[:, 2])
        try:
            abcd = rhs @ np.linalg.inv(coefficient_matrix)
        except np.linalg.LinAlgError:
            abcd = rhs @ np.linalg.inv(coefficient_matrix + 1e-5 * np.identity(2))
        a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
        direction = np.array([a, 1, c]) / np.linalg.norm(np.array([a, 1, c]))
        y_positive = np.dot(direction, np.array([0, 1, 0])) > 0
        if y_positive:
            y_min = np.min(points[:, 1])
            pivot = np.array([a * y_min + b, y_min, c * y_min + d])
        else:
            y_max = np.max(points[:, 1])
            pivot = np.array([a * y_max + b, y_max, c * y_max + d])
    else:
        # fit line, x=az+b y=cz+d
        coefficient_matrix = np.zeros((2, 2))
        coefficient_matrix[0, 0] = np.sum(points[:, 2]**2)
        coefficient_matrix[0, 1] = np.sum(points[:, 2])
        coefficient_matrix[1, 0] = coefficient_matrix[0, 1]
        coefficient_matrix[1, 1] = N
        rhs = np.zeros((2, 2))
        rhs[0, 0] = np.sum(points[:, 0] * points[:, 2])
        rhs[0, 1] = np.sum(points[:, 0])
        rhs[1, 0] = np.sum(points[:, 1] * points[:, 2])
        rhs[1, 1] = np.sum(points[:, 1])
        try:
            abcd = rhs @ np.linalg.inv(coefficient_matrix)
        except np.linalg.LinAlgError:
            abcd = rhs @ np.linalg.inv(coefficient_matrix + 1e-5 * np.identity(2))
        a, b, c, d = abcd[0, 0], abcd[0, 1], abcd[1, 0], abcd[1, 1]
        direction = np.array([a, c, 1]) / np.linalg.norm(np.array([a, c, 1]))
        z_positive = np.dot(direction, np.array([0, 0, 1])) > 0
        if z_positive:
            z_min = np.min(points[:, 2])
            pivot = np.array([a * z_min + b, c * z_min + d, z_min])
        else:
            z_max = np.max(points[:, 2])
            pivot = np.array([a * z_max + b, c * z_max + d, z_max])
    return (pivot, direction)


def refit(fit_C:np.ndarray, fit_A:np.ndarray, ref_C:np.ndarray, ref_A:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # min ||fit_C + t * fit_A - ref_C||^2 = min fit_A^T fit_A t^2 + 2 fit_A^T (fit_C - ref_C) t + (fit_C - ref_C)^T (fit_C - ref_C)
    if np.dot(fit_A, ref_A) < 0:
        refit_A = -np.copy(fit_A)
    else:
        refit_A = np.copy(fit_A)
    t = -1 * np.dot(refit_A, fit_C - ref_C) / np.dot(refit_A, refit_A)
    refit_C = fit_C + t * refit_A
    return (refit_C, refit_A)


if __name__ == '__main__':
    points = np.array([[-0.3, 0.46, 0.83],
                    [-0.2, 0.254, 0.946],
                    [-0.1, 0.111, 0.989],
                    [0, 0, 1],
                    [0.1, -0.0910, 0.991],
                    [0.2, -0.166, 0.966],
                    [0.3, -0.227, 0.927],
                    [0.4, -0.275, 0.874],
                    [0.5, -0.309, 0.809],
                    [0.6, -0.329, 0.729],
                    [0.7, -0.332, 0.632],
                    [0.8, -0.312, 0.512],
                    [0.9, -0.254, 0.354],
                    [0.8, 0.512, -0.312],
                    [0.7, 0.632, -0.332],
                    [0.6, 0.729, -0.329],
                    [0.5, 0.809, -0.309],
                    [0.4, 0.874, -0.274],
                    [0.3, 0.927, -0.227],
                    [0.2, 0.966, -0.166],
                    [0.1, 0.991, -0.091],
                    [0, 1, 0],
                    [-0.1, 0.989, 0.111],
                    [-0.2, 0.946, 0.254],
                    [-0.3, 0.83, 0.45]])
    points = np.array([[11.5713, 6.9764, 10.4685],
                    [11.5859, 9.088, 13.5831],
                    [11.5802, 11.1949, 14.6103],
                    [11.5542, 13.312, 14.7279],
                    [11.5692, 15.3806, 14.0576],
                    [11.5632, 17.4873, 12.1397],
                    [11.5598, 17.8894, 6.4025],
                    [11.5577, 15.8714, 4.0703],
                    [11.5729, 13.8578, 3.232],
                    [11.5657, 11.8711, 3.1326],
                    [11.5706, 9.8797, 3.7866],
                    [11.5663, 7.8676, 5.5348]])
    C, A = fit_arc(points)

    import open3d as o3d
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)
    
    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1)
    joint.paint_uniform_color([1, 0, 0])
    rotation = np.zeros((3, 3))
    temp2 = np.cross(A, np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-6:
        temp1 = np.cross(np.array([0., 1., 0.]), A)
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(A, temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, A)
        temp1 /= np.linalg.norm(temp1)
    rotation[:, 0] = temp1
    rotation[:, 1] = temp2
    rotation[:, 2] = A
    joint.rotate(rotation, np.array([[0], [0], [0]]))
    joint.translate(C.reshape((3, 1)))
    geometries.append(joint)

    o3d.visualization.draw_geometries(geometries)

    points = np.array([[2., 2., 1.],
                       [2.5, 2.5, 1.5],
                       [3., 3., 2.],
                       [3.5, 3.5, 2.5],
                       [4., 4., 3.],
                       [4.5, 4.5, 3.5],
                       [5., 5., 4.],
                       [5.5, 5.5, 4.5]])
    points = np.array([[2., 2., 1.],
                       [2.5, 2.5, 1.6],
                       [3.2, 3.1, 1.9],
                       [3.5, 3.6, 2.5],
                       [3.8, 4., 3.1],
                       [4.5, 4.6, 3.6],
                       [5., 5., 4.],
                       [5.51, 5.48, 4.6]])
    pivot, direction = fit_line(points)

    import open3d as o3d
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)

    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.2, cone_height=0.1)
    joint.paint_uniform_color([1, 0, 0])
    rotation = np.zeros((3, 3))
    temp2 = np.cross(direction, np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-6:
        temp1 = np.cross(np.array([0., 1., 0.]), direction)
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(direction, temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, direction)
        temp1 /= np.linalg.norm(temp1)
    rotation[:, 0] = temp1
    rotation[:, 1] = temp2
    rotation[:, 2] = direction
    joint.rotate(rotation, np.array([[0], [0], [0]]))
    joint.translate(pivot.reshape((3, 1)))
    geometries.append(joint)

    o3d.visualization.draw_geometries(geometries)
