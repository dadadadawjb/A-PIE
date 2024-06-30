from sensor_msgs.msg import PointCloud2,PointField
import rospy
from graspnet_pkg.srv import GraspNetList,GraspNetListResponse
from msg_srv.srv import Affordance
from msg_srv.srv import PPF
import numpy as np
from pyquaternion import Quaternion as Quat
import open3d as o3d


def xyzl_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
        Numpy to PointCloud2
        Create a sensor_msgs.PointCloud2 from an array
        of points (x, y, z, l)
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            # PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg


def getGraspNetService(pc_msg):
    rospy.wait_for_service("/GraspNet")
    try:
        srv_handle = rospy.ServiceProxy("/GraspNet", GraspNetList)
        rep = srv_handle(pc_msg)
        GraspMsg_list = rep.gg

    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
        return None
    
    return GraspMsg_list


def getAxisPPFService(pc_msg):
    rospy.wait_for_service("/ppf_service")
    try:
        srv_handle = rospy.ServiceProxy("/ppf_service",PPF)
        result = srv_handle(pointcloud=pc_msg)
    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
    # len(axis) = n*axis[6]
    return result.axis


def getAffordanceService(pc_msg,GraspMsg_list,axis):
    rospy.wait_for_service("affordance_service")
    try:
        srv_handle = rospy.ServiceProxy("affordance_service",Affordance)
        result = srv_handle(pointcloud=pc_msg,gg=GraspMsg_list,axis=axis)
    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
    # axis n*axis[6]
    return result.result


def convertGraspMsgtoNumpy(gg_list):
    # grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids
    gg_array = []
    for gg in gg_list:
        grasp_score = gg.grasp_score
        grasp_width = gg.grasp_width
        grasp_height = gg.grasp_height
        grasp_depth = gg.grasp_depth
        pre_ = [grasp_score, grasp_width, grasp_height, grasp_depth]
        rotation = gg.rotation
        grasp_center = gg.grasp_center
        obj_ids = gg.obj_ids
        quat = Quat(rotation.w, rotation.x, rotation.y, rotation.z)
        rotation_matrix = quat.rotation_matrix.flatten().tolist()
        grasp_center = [grasp_center.x, grasp_center.y, grasp_center.z]
        gg_array.append(pre_ + rotation_matrix + grasp_center + [obj_ids])
    gg_array = np.array(gg_array)
    return gg_array


if __name__ == "__main__":
    file_path = '0000.npy'
    pointcloud = np.load(file_path)
    
    pc_msg = xyzl_array_to_pointcloud2(pointcloud,frame_id='pc_base')
    axis = getAxisPPFService(pc_msg)
    
    GraspMsg_list = getGraspNetService(pc_msg)
    
    num = int(len(axis)/6)
    axises = []
    axis = list(axis)
    for i in range(num):
        axises.extend(axis[i*6:i*6+6] + [0])
    result = getAffordanceService(pc_msg,GraspMsg_list,axises) 
    grasps = convertGraspMsgtoNumpy(GraspMsg_list)
	# grasp_group = GraspGroup(gg)
    axis = np.array(axis)
    # visualize
    for j in range(num):
        geometries = []

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(frame)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        geometries.append(pcd)

        joint = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
        rotation = np.zeros((3, 3))
        temp2 = np.cross(axis[j*6+3:j*6+6], np.array([1., 0., 0.]))
        if np.linalg.norm(temp2) < 1e-6:
            temp1 = np.cross(np.array([0., 1., 0.]), axis[j*6+3:j*6+6])
            temp1 /= np.linalg.norm(temp1)
            temp2 = np.cross(axis[j*6+3:j*6+6], temp1)
            temp2 /= np.linalg.norm(temp2)
        else:
            temp2 /= np.linalg.norm(temp2)
            temp1 = np.cross(temp2, axis[j*6+3:j*6+6])
            temp1 /= np.linalg.norm(temp1)
        rotation[:, 0] = temp1
        rotation[:, 1] = temp2
        rotation[:, 2] = axis[j*6+3:j*6+6]
        joint.rotate(rotation, np.array([[0], [0], [0]]))
        joint.translate(axis[j*6:j*6+3].reshape((3, 1)))
        geometries.append(joint)

        finger_width = 0.004
        tail_length = 0.04
        depth_base = 0.02
        for i in range(grasps.shape[0]):
            gg = grasps[i]
            gg_affordance = result[j].floatlist[i]
            gg_score = gg[0]
            gg_width = gg[1]
            gg_depth = gg[3]
            gg_translation = gg[13:16]
            gg_rotation = gg[4:13].reshape((3, 3))

            left = np.zeros((2, 3))
            left[0] = np.array([-depth_base - finger_width, -gg_width / 2, 0])
            left[1] = np.array([gg_depth, -gg_width / 2, 0])

            right = np.zeros((2, 3))
            right[0] = np.array([-depth_base - finger_width, gg_width / 2, 0])
            right[1] = np.array([gg_depth, gg_width / 2, 0])

            bottom = np.zeros((2, 3))
            bottom[0] = np.array([-finger_width - depth_base, -gg_width / 2, 0])
            bottom[1] = np.array([-finger_width - depth_base, gg_width / 2, 0])

            tail = np.zeros((2, 3))
            tail[0] = np.array([-(tail_length + finger_width + depth_base), 0, 0])
            tail[1] = np.array([-(finger_width + depth_base), 0, 0])

            vertices = np.vstack([left, right, bottom, tail])
            vertices = np.dot(gg_rotation, vertices.T).T + gg_translation

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1], [2, 3], [4, 5], [6, 7]])
            if gg_affordance < 0.5:
                line_set.paint_uniform_color([1, 2*gg_affordance, 0])
            else:
                line_set.paint_uniform_color([-2*gg_affordance+2, 1, 0])
            geometries.append(line_set)
        
        o3d.visualization.draw_geometries(geometries, window_name=f"joint {j}")
    
    