from frankx import Affine, JointMotion, Robot, Waypoint, WaypointMotion, Gripper, LinearRelativeMotion
import numpy as np
import pdb


class Panda():
    def __init__(self,host='172.16.0.2'):
        self.robot = Robot(host)
        self.gripper = Gripper(host)
        self.setGripper(20,0.1)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        # Reduce the acceleration and velocity dynamic
        self.robot.set_dynamic_rel(0.2)
        # self.robot.set_dynamic_rel(0.05)

        # self.robot.velocity_rel = 0.1
        # self.robot.acceleration_rel = 0.02
        # self.robot.jerk_rel = 0.01

        self.joint_tolerance = 0.01
        # state = self.robot.read_once()
        # print('\nPose: ', self.robot.current_pose())
        # print('O_TT_E: ', state.O_T_EE)
        # print('Joints: ', state.q)
        # print('Elbow: ', state.elbow)
        # pdb.set_trace()
    def setGripper(self,force=20.0, speed=0.02):
        self.gripper.gripper_speed = speed # m/s
        self.gripper.gripper_force = force # N

    def gripper_close(self):
        self.gripper.clamp()
    
    def gripper_open(self):
        self.gripper.open()
        
    def move_gripper(self, width):
        self.gripper.move(width, self.gripper.gripper_speed)

    def moveJoint(self,joint,moveTarget=True):
        assert len(joint)==7, "panda DOF is 7"
        self.robot.move(JointMotion(joint))
        # while moveTarget:
        #     current = self.robot.read_once().q
        #     if all([np.abs(current[j] - joint[j])<self.joint_tolerance for j in range(len(joint))]):
        #         break
        return True


    def moveRelativePose(self,pose):
        if len(pose) == 6:
            motion = LinearRelativeMotion(Affine(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]))
        elif len(pose) == 3:
            motion = LinearRelativeMotion(Affine(pose[0], pose[1], pose[2]))
        else:
            raise ValueError
        self.robot.move(motion)

    def moveWaypoints(self,pose_list=None):
        '''

        :param pose_list: [pose], pose =[x,y,z,R,P,Y,elbow_q]
        :return:
        '''
        waypointList = []
        for pose in pose_list:
            waypoint = Waypoint(Affine(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]), pose[6], Waypoint.Absolute)
            waypointList.append(waypoint)
        motion_down = WaypointMotion(waypointList)
        # # waypoint1 = Waypoint(Affine(0.439397, 0.193227, 0.445747, 0.443057, -0.455486, 2.924881), 0.5019,Waypoint.Absolute)
        # # waypoint2 = Waypoint(Affine(0.768568, -0.111737, 0.323822, -0.017118, -0.957058, 2.965727), -0.2627,Waypoint.Absolute)
        # pose1 = Waypoint(Affine(0.519510, -0.055463, 0.392558, -0.207360, 0.138675, -3.056701), -1.374,Waypoint.Absolute)
        # pose2 = Waypoint(Affine(0.811605, -0.098900, 0.314441, -0.218991, -0.521403, -2.956528), -2.739,Waypoint.Absolute)
        # motion_down = WaypointMotion([
        #     pose1,pose2])
        #
        # # You can try to block the robot now.
        # pdb.set_trace()
        self.robot.move(motion_down)


if __name__ == "__main__":
    robot = Panda()
    joint2 = [1.182225251800949, -0.5273177195765826, -1.2623061685896755, -2.0766707371159603, -0.2919127081429459,
              1.6921455342287042, 0.7338029986078157]
    joint1 = [2.0489815386924866, -0.5843167256389248, -2.0282640731330166, -1.9333003799712163, -0.29245471532707795,
              3.009355127943887, 0.5171093538651864]
    # robot.moveJoint(joint1)
    # robot.moveJoint(joint2)
    pose1 = [0.519510, -0.055463, 0.392558, -0.207360, 0.138675, -3.056701, -1.374]
    pose2 = [0.811605, -0.098900, 0.314441, -0.218991, -0.521403, -2.956528, -2.739]
    # robot.moveWaypoints([pose1,pose2])
    pdb.set_trace()
