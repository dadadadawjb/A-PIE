from typing import Tuple, List, Optional
import numpy as np
import transformations as tf
import pybullet as pb


def get_depth(z_n:float, zNear:float, zFar:float) -> float:
    z_n = 2.0 * z_n - 1.0
    z_e = 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear))
    return z_e

def get_point_cloud(width:int, height:int, view_matrix:Tuple[float], proj_matrix:Tuple[float], 
                    obj_id:int) -> np.ndarray:
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    # get a depth image
    # "infinite" depths will have a value close to 1
    image_arr = pb.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    depth = image_arr[3]
    seg = image_arr[4]
    seg_object = (seg == obj_id).reshape(-1)

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[np.bitwise_and(z < 0.99,seg_object)]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points


def get_viewmat_and_extrinsic(cameraEyePosition:List[float], cameraTargetPosition:List[float], 
                              cameraUpVector:List[float]) -> Tuple[Tuple[float], np.ndarray]:
    view_matrix = pb.computeViewMatrix(
        cameraEyePosition=cameraEyePosition,
        cameraTargetPosition=cameraTargetPosition,
        cameraUpVector=cameraUpVector
    )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, f'cameraTargetPosition and cameraEyePosition is at same location'
    z /= norm

    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, f'cameraUpVector is parallel to z axis'
    y /= norm

    x = np.cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return (view_matrix, extrinsic)

def get_projmat_and_intrinsic(width:int, height:int, fx:float, fy:float, 
                              near:float, far:float) -> Tuple[Tuple[float], np.ndarray]:
    cx = width / 2
    cy = height / 2
    fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

    project_matrix = pb.computeProjectionMatrixFOV(
        fov=fov,
        aspect=1.0,
        nearVal=near,
        farVal=far
    )

    intrinsic = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])

    return (project_matrix, intrinsic)

def get_camera(target_x:List[float], target_y:List[float], target_z:List[float], 
    distance:List[float], yaw:List[float], pitch:List[float], roll:List[float], up_axis:List[str], 
    fov:List[float], nearVal:List[float], farVal:List[float], height:List[int], width:List[int]) -> Tuple[int, List[Tuple[float]], List[Tuple[float]], List[np.ndarray]]:
    assert len(target_x) == len(target_y) == len(target_z) \
        == len(distance) == len(yaw) == len(roll) == len(up_axis) \
        == len(fov) == len(nearVal) == len(farVal) == len(height) == len(width)
    num_camera = len(nearVal)
    for i in range(num_camera):
        if up_axis[i] == 'y':
            up_axis[i] = 1
        elif up_axis[i] == 'z':
            up_axis[i] = 2
        else:
            raise ValueError
    
    view_matrix = []
    proj_matrix = []
    extrinsic = []
    for i in range(num_camera):
        view_matrix.append(pb.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[target_x[i], target_y[i], target_z[i]], 
                                                        distance=distance[i], 
                                                        yaw=yaw[i], 
                                                        pitch=pitch[i], 
                                                        roll=roll[i], 
                                                        upAxisIndex=up_axis[i]))
        proj_matrix.append(pb.computeProjectionMatrixFOV(fov=fov[i], 
                                                    aspect=width[i]/height[i], 
                                                    nearVal=nearVal[i], 
                                                    farVal=farVal[i]))
        rot_mat = np.array(pb.getMatrixFromQuaternion(pb.getQuaternionFromEuler(
                        [pitch[i] / 180. * np.pi, roll[i] / 180. * np.pi, yaw[i] / 180. * np.pi]))).reshape(3, 3)
        ext_mat = np.identity(4)        # extrinsic convention: x right, y down, z in
        if up_axis[i] == 1:             # y up, z out, x right
            ext_mat[:3, 0] = rot_mat[:, 0]
            ext_mat[:3, 1] = -rot_mat[:, 1]
            ext_mat[:3, 2] = -rot_mat[:, 2]
            ext_mat[:3, 3] = rot_mat[:, 2] * distance[i] + np.array([target_x[i], target_y[i], target_z[i]])
        elif up_axis[i] == 2:           # z up, y in, x right
            ext_mat[:3, 0] = rot_mat[:, 0]
            ext_mat[:3, 1] = -rot_mat[:, 2]
            ext_mat[:3, 2] = rot_mat[:, 1]
            ext_mat[:3, 3] = -rot_mat[:, 1] * distance[i] + np.array([target_x[i], target_y[i], target_z[i]])
        else:
            raise ValueError
        extrinsic.append(ext_mat)
    return (num_camera, view_matrix, proj_matrix, extrinsic)

def generate_camera(num:int, object_id:int, camera_distance_min:float, camera_distance_max:float, 
    fov:float, width:int, height:int, near:float, far:float, 
    cone_direction:np.ndarray, cone_angle:float, up_axis:np.ndarray) -> Tuple[List[Tuple[float]], List[Tuple[float]], List[np.ndarray]]:
    object_pos, object_ori = pb.getBasePositionAndOrientation(object_id)
    object_pos = np.array(object_pos)
    object_ori = np.array(object_ori)
    object_ori = pb.getMatrixFromQuaternion(object_ori)
    object_ori = np.array(object_ori).reshape((3, 3))

    view_matrix = []
    proj_matrix = []
    extrinsic = []
    for _ in range(num):
        z = np.random.uniform(np.cos(cone_angle), 1)
        theta = np.random.uniform(0, 2 * np.pi)
        upright_cone = np.array([np.sqrt(1 - z**2) * np.cos(theta), np.sqrt(1 - z**2) * np.sin(theta), z])
        cone_rot_axis = np.cross(np.array([0, 0, 1]), cone_direction)
        cone_rot_axis = cone_rot_axis / np.linalg.norm(cone_rot_axis)
        cone_rot_angle = np.arccos(np.dot(np.array([0, 0, 1]), cone_direction))
        cone_rot = tf.rotation_matrix(cone_rot_angle, cone_rot_axis, [0, 0, 0])[:3, :3]
        real_cone = np.dot(cone_rot, upright_cone)
        camera_ori = np.dot(object_ori, real_cone)
        camera_distance = np.random.uniform(camera_distance_min, camera_distance_max)
        camera_pos = object_pos + camera_ori * camera_distance
        cos_theta = np.dot(real_cone, up_axis)
        if cos_theta > 0:
            camera_up = 1/cos_theta * up_axis - real_cone
        elif cos_theta < 0:
            camera_up = -1 * (1/cos_theta * up_axis - real_cone)
        else:
            camera_up = up_axis
        camera_up = camera_up / np.linalg.norm(camera_up)
        camera_up = np.dot(object_ori, camera_up)

        view_mat = pb.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=object_pos, cameraUpVector=camera_up)
        proj_mat = pb.computeProjectionMatrixFOV(fov=fov, aspect=width/height, nearVal=near, farVal=far)
        view_matrix.append(view_mat)
        proj_matrix.append(proj_mat)

        ext_mat = np.identity(4)        # extrinsic convention: x right, y down, z in
        ext_mat[:3, 2] = (object_pos - camera_pos) / np.linalg.norm(object_pos - camera_pos)
        ext_mat[:3, 1] = -camera_up
        ext_mat[:3, 0] = np.cross(ext_mat[:3, 1], ext_mat[:3, 2])
        ext_mat[:3, 3] = camera_pos
        extrinsic.append(ext_mat)
    return (view_matrix, proj_matrix, extrinsic)


def get_base_pose(body_id:int) -> np.ndarray:
    pos, ori = pb.getBasePositionAndOrientation(body_id)
    R_mat = np.array(pb.getMatrixFromQuaternion(ori)).reshape(3,3)
    pose = np.identity(4)
    pose[:3,:3] = R_mat
    pose[:3,3] = np.array(pos)
    return pose

def get_link_pos(body_id:int, link_id:int) -> np.ndarray:
    if link_id == -1:
        pos, ori = pb.getBasePositionAndOrientation(body_id)
    else:
        pos, ori = pb.getLinkState(body_id, link_id)[4:6]
    R_Mat = np.array(pb.getMatrixFromQuaternion(ori)).reshape(3, 3)
    pose = np.identity(4)
    pose[:3,:3] = R_Mat
    pose[:3, 3] = np.array(pos)
    return pose

def get_joint_poses(body_id:int, link_id:List[int]) -> List[np.ndarray]:
    joint_poses = []
    for i in range(len(link_id)):
        joint_pose = get_link_pos(body_id, link_id[i])
        joint_poses.append(joint_pose)
    return joint_poses


def draw_link_coord(pose:Optional[np.ndarray]=None, body_id:Optional[int]=None, link_id:Optional[int]=None, 
                    length:float=0.1, width:float=3.) -> Tuple[int, int, int]:
    if pose is not None:
        frame_start_postition = pose[:3,3]
        R_Mat = pose[:3,:3]
    else:
        frame_start_postition, frame_posture = pb.getLinkState(body_id, link_id)[4:6]
        R_Mat = np.array(pb.getMatrixFromQuaternion(frame_posture)).reshape(3, 3)

    x_axis = R_Mat[:, 0]
    x_end_p = (np.array(frame_start_postition) + np.array(x_axis * length)).tolist()
    x_line_id = pb.addUserDebugLine(frame_start_postition, x_end_p, [1, 0, 0], lineWidth=width)

    y_axis = R_Mat[:, 1]
    y_end_p = (np.array(frame_start_postition) + np.array(y_axis * length)).tolist()
    y_line_id = pb.addUserDebugLine(frame_start_postition, y_end_p, [0, 1, 0], lineWidth=width)

    z_axis = R_Mat[:, 2]
    z_end_p = (np.array(frame_start_postition) + np.array(z_axis * length)).tolist()
    z_line_id = pb.addUserDebugLine(frame_start_postition, z_end_p, [0, 0, 1], lineWidth=width)

    return (x_line_id, y_line_id, z_line_id)

def draw_gripper(center_cam:np.ndarray, R_cam:np.ndarray, width:float, depth:float, 
                 extrinsic:np.ndarray, score:float=1, 
                 color:Optional[List[float]]=None, coord_id:Optional[List[int]]=None) -> Tuple[int, int, int, int]:
    line_width = 3.
    finger_width = 0.004
    tail_length = 0.04
    depth_base = 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score
    left = np.zeros((2, 3))
    left[0] = np.array([-depth_base - finger_width, -width / 2, 0])
    left[1] = np.array([depth, -width / 2, 0])

    right = np.zeros((2, 3))
    right[0] = np.array([-depth_base - finger_width, width / 2, 0])
    right[1] = np.array([depth, width / 2, 0])

    bottom = np.zeros((2, 3))
    bottom[0] = np.array([-finger_width - depth_base, -width / 2, 0])
    bottom[1] = np.array([-finger_width - depth_base, width / 2, 0])

    tail = np.zeros((2, 3))
    tail[0] = np.array([-(tail_length + finger_width + depth_base), 0, 0])
    tail[1] = np.array([-(finger_width + depth_base), 0, 0])

    vertices = np.vstack([left, right, bottom, tail])
    vertices = np.dot(R_cam, vertices.T).T + center_cam
    vertices = np.dot(extrinsic[:3,:3], vertices.T).T + extrinsic[:3,3]

    # left
    kwargs = dict()
    kwargs['lineWidth'] = line_width
    kwargs['lineColorRGB'] = [color_r, color_g, color_b]
    if coord_id is not None:
        kwargs['replaceItemUniqueId'] = coord_id[0]
    left_id = pb.addUserDebugLine(vertices[0], vertices[1], **kwargs)

    # right
    if coord_id is not None:
        kwargs['replaceItemUniqueId'] = coord_id[1]
    right_id = pb.addUserDebugLine(vertices[2], vertices[3], **kwargs)

    # bottom
    if coord_id is not None:
        kwargs['replaceItemUniqueId'] = coord_id[2]
    bottom_id = pb.addUserDebugLine(vertices[4], vertices[5], **kwargs)

    # tail
    if coord_id is not None:
        kwargs['replaceItemUniqueId'] = coord_id[3]
    tail_id = pb.addUserDebugLine(vertices[6], vertices[7], **kwargs)

    return (left_id, right_id, bottom_id, tail_id)
