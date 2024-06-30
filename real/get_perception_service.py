from sensor_msgs.msg import PointCloud2, PointField
from msg_srv.srv import PPF
import numpy as np
import rospy
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


def get_axis_ppf(pc_msg):
    rospy.wait_for_service("ppf_service")
    try:
        srv_handle = rospy.ServiceProxy("ppf_service",PPF)
        result = srv_handle(pointcloud=pc_msg)
    except rospy.ServiceException as e:
        print("Service call failed : %s" % e)
    return result.axis


if __name__ == "__main__":
    file_path = '0000.npy'
    pointcloud = np.load(file_path)
    
    pc_msg = xyzl_array_to_pointcloud2(pointcloud)
    axis = get_axis_ppf(pc_msg)
    
    # visualize
    geometries = []
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    geometries.append(frame)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    geometries.append(pcd)
    num = int(len(axis)/6)
    for j in range(num):
        direction = np.asarray(axis[j*6+3:j*6+6])
        translation = np.asarray(axis[j*6:j*6+3])
        joint_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.4, cone_height=0.1, resolution=20, cylinder_split=4, cone_split=1)
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
        joint_axis.rotate(rotation, np.array([[0], [0], [0]]))
        joint_axis.translate(translation.reshape((3, 1)))
        joint_axis.paint_uniform_color([204/255, 204/255, 204/255])
        geometries.append(joint_axis)
        joint_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        joint_point = joint_point.translate(translation.reshape((3, 1)))
        joint_point.paint_uniform_color([179/255, 179/255, 179/255])
        geometries.append(joint_point)
    o3d.visualization.draw_geometries(geometries)

    