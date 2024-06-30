import numpy as np
import torch
import transformations as tf

def generate_arc_grad(translation:torch.Tensor, direction:torch.Tensor, start:torch.Tensor, degree:float, num:int) -> torch.Tensor:
    # transform from original frame to joint unit earth frame, 
    # translation as origin, direction as z axis, start-translation lies in yz plane with unit length
    # translation: (3,), direction: (3,), start: (3,)
    rotation_matrix = torch.eye(3, device=translation.device)
    z_axis = direction
    y_axis = start - translation
    x_axis = torch.cross(y_axis, z_axis)
    y_axis = torch.cross(z_axis, x_axis)
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = y_axis / torch.norm(y_axis)
    z_axis = z_axis / torch.norm(z_axis)
    rotation_matrix[:3, 2] = z_axis
    rotation_matrix[:3, 1] = y_axis
    rotation_matrix[:3, 0] = x_axis
    scale_matrix = torch.norm(start - translation) * torch.eye(3, device=translation.device)
    transform_matrix = torch.eye(4, device=translation.device)
    transform_matrix[:3, 3] = translation
    transform_matrix[:3, :3] = torch.mm(rotation_matrix, scale_matrix)
    transform_matrix = torch.inverse(transform_matrix)

    transformed_start = torch.mm(transform_matrix[:3, :3], start[:, None])[:, 0] + transform_matrix[:3, 3]

    # generate arc
    arc_rotation_matrix = tf.rotation_matrix(angle=-degree / 180.0 * np.pi, direction=[0, 0, 1], point=[0, 0, 0])
    arc_rotation_matrix = torch.from_numpy(arc_rotation_matrix[:3, :3]).to(translation.device).to(translation.dtype)
    arc = torch.zeros((num, 3), device=translation.device, dtype=translation.dtype)
    for i in range(num):
        arc[i] = torch.mm(arc_rotation_matrix, transformed_start[:, None])[:, 0]
        transformed_start = arc[i]
    
    # transform from joint unit earth frame to original frame
    transform_matrix_inv = torch.inverse(transform_matrix)
    arc = torch.mm(arc, transform_matrix_inv[:3, :3].t()) + transform_matrix_inv[:3, 3]
    return arc

def generate_line_grad(translation:torch.Tensor, direction:torch.Tensor, start:torch.Tensor, step:float, num:int) -> torch.Tensor:
    # transform from original frame to joint directional frame, 
    # translation as origin, direction as z axis, start lies on origin
    # translation: (3,), direction: (3,), start: (3,)
    transformed_start = torch.tensor([0, 0, 0], device=translation.device, dtype=translation.dtype)

    # generate line
    step_vector = torch.tensor([0, 0, step/100.0], device=translation.device, dtype=translation.dtype)
    transformed_line = torch.zeros((num, 3), device=translation.device, dtype=translation.dtype)
    for i in range(num):
        transformed_line[i] = transformed_start + step_vector
        transformed_start = transformed_line[i]
    
    # transform from joint directional frame to original frame
    line = torch.zeros((num, 3), device=translation.device, dtype=translation.dtype)
    for i in range(num):
        line[i] = torch.norm(transformed_line[i]) * direction + start
    return line


if __name__ == '__main__':
    translation = torch.tensor([1, 1, 0], dtype=torch.float32, requires_grad=True)
    direction = torch.tensor([0, 1, 0], dtype=torch.float32, requires_grad=True)
    start = torch.tensor([2, 2, 1], dtype=torch.float32)
    arc = generate_arc_grad(translation, direction, start, 5, 5)
    line = generate_line_grad(translation, direction, start, 10, 5)

    import torch.nn as nn
    label_arc = torch.zeros((5, 3), dtype=torch.float32)
    loss = nn.MSELoss()
    loss_value = loss(arc, label_arc)
    loss_value.backward()
    label_line = torch.zeros((5, 3), dtype=torch.float32)
    loss = nn.MSELoss()
    loss_value = loss(line, label_line)
    loss_value.backward()

    import open3d as o3d
    geometries = []
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    geometries.append(frame)
    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.008, cone_radius=0.02, cylinder_height=0.2, cone_height=0.05)
    rotation = np.zeros((3, 3))
    temp2 = np.cross(direction.detach().numpy(), np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-6:
        temp1 = np.cross(np.array([0., 1., 0.]), direction.detach().numpy())
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(direction.detach().numpy(), temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, direction.detach().numpy())
        temp1 /= np.linalg.norm(temp1)
    rotation[:, 0] = temp1
    rotation[:, 1] = temp2
    rotation[:, 2] = direction.detach().numpy()
    joint.rotate(rotation, np.array([[0], [0], [0]]))
    joint.translate(translation.detach().numpy().reshape((3, 1)))
    joint.paint_uniform_color([0, 0, 1])
    geometries.append(joint)
    start_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_point.translate(start.numpy().reshape((3, 1)))
    start_point.paint_uniform_color([1, 0, 0])
    geometries.append(start_point)
    for i in range(arc.shape[0]):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        point.translate(arc[i].detach().numpy().reshape((3, 1)))
        point.paint_uniform_color([0, 1, 0])
        geometries.append(point)
    o3d.visualization.draw_geometries(geometries)

    geometries = []
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    geometries.append(frame)
    joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.008, cone_radius=0.02, cylinder_height=0.2, cone_height=0.05)
    rotation = np.zeros((3, 3))
    temp2 = np.cross(direction.detach().numpy(), np.array([1., 0., 0.]))
    if np.linalg.norm(temp2) < 1e-6:
        temp1 = np.cross(np.array([0., 1., 0.]), direction.detach().numpy())
        temp1 /= np.linalg.norm(temp1)
        temp2 = np.cross(direction.detach().numpy(), temp1)
        temp2 /= np.linalg.norm(temp2)
    else:
        temp2 /= np.linalg.norm(temp2)
        temp1 = np.cross(temp2, direction.detach().numpy())
        temp1 /= np.linalg.norm(temp1)
    rotation[:, 0] = temp1
    rotation[:, 1] = temp2
    rotation[:, 2] = direction.detach().numpy()
    joint.rotate(rotation, np.array([[0], [0], [0]]))
    joint.translate(translation.detach().numpy().reshape((3, 1)))
    joint.paint_uniform_color([0, 0, 1])
    geometries.append(joint)
    start_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_point.translate(start.numpy().reshape((3, 1)))
    start_point.paint_uniform_color([1, 0, 0])
    geometries.append(start_point)
    for i in range(line.shape[0]):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        point.translate(line[i].detach().numpy().reshape((3, 1)))
        point.paint_uniform_color([0, 1, 0])
        geometries.append(point)
    o3d.visualization.draw_geometries(geometries)
