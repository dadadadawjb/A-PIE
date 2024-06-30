import pybullet as pb
import os
import numpy as np
from collections import namedtuple
import math
from scipy.spatial.transform import Rotation as Rot


description_path = os.path.dirname(
    os.path.abspath(__file__)) + "/objects/robotiq_85.urdf"

floating_description_path = os.path.dirname(
    os.path.abspath(__file__)) + "/objects/robotiq_85_combined.urdf"

floating_pad_gripper_description_path = os.path.dirname(
    os.path.abspath(__file__)) + "/objects/pad_gripper_visual.urdf"  # pad_gripper_visual.urdf, pad_gripper.urdf


class Robotiq85Gripper():

    def __init__(self,
                 robot_description=description_path,
                 useFixedBase=True,
                 uid=None,
                 gravity=False,
                 realtime_sim=False):
        """
        :param robot_description: path to description file (urdf, .bullet, etc.)
        :param config: optional config file for specifying robot information
        :param uid: optional server id of bullet

        :type robot_description: str
        :type config: dict
        :type uid: int
        """
        self._ready = False
        if uid is None:
            uid = pb.connect(pb.GUI)

        self._uid = uid
        pb.resetSimulation(physicsClientId=self._uid)
        robot_id = self.load_model(robot_description, useFixedBase)
        self._id = robot_id

        if gravity:
            pb.setGravity(0.0, 0.0, -9.8, physicsClientId=self._uid)
        else:
            pb.setGravity(0.0, 0.0, 0., physicsClientId=self._uid)

        self.gravity = gravity

        if realtime_sim:
            pb.setRealTimeSimulation(1, physicsClientId=self._uid)
            pb.setTimeStep(0.01, physicsClientId=self._uid)
        self._rt_sim = realtime_sim

        self.gripper_range = [0, 0.085]

        # by default, set FT sensor at last fixed joint
        # self._ft_joints = [0]
        # self.set_ft_sensor_at(self._ft_joints[0])
        self.__parse_joint_info__()
        self.__post_load__()

        self._all_joints = np.array(range(pb.getNumJoints(self._id, physicsClientId=self._uid)))
        self._movable_joints = np.array(self.controllable_joints)

        self._ready = True

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = pb.createConstraint(self._id, self.mimic_parent_id,
                                   self._id, joint_id,
                                   jointType=pb.JOINT_GEAR,
                                   jointAxis=[1, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            pb.changeConstraint(c, gearRatio=-multiplier, maxForce=10000, erp=1)  # Note: the mysterious `erp` is of EXTREME importance


    def __parse_joint_info__(self):
        numJoints = pb.getNumJoints(self._id)
        jointInfo = namedtuple('jointInfo',['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                                'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = pb.getJointInfo(self._id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != pb.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                pb.setJointMotorControl2(self._id, jointID, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

    def set_ft_sensor_at(self, joint_id, enable=True):
        print("FT sensor at joint", joint_id)
        pb.enableJointForceTorqueSensor(self._id, joint_id, enable, self._uid)


    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        pb.setJointMotorControl2(self._id, self.mimic_parent_id, pb.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        [pb.stepSimulation() for i in range(10)]


    # def hold_gripper(self,force=600):
    #     for i in [1, 2]:
    #         pb.setJointMotorControl2(self._id, i, pb.TORQUE_CONTROL, force=force)
    #     [pb.stepSimulation() for _ in range(10)]

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def load_model(self,robot_description, useFixedBase):
        extension = robot_description.split('.')[-1]
        if extension == "urdf":
            robot_id = pb.loadURDF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        elif extension == 'sdf':
            robot_id = pb.loadSDF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        elif extension == 'bullet':
            robot_id = pb.loadBullet(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        else:
            robot_id = pb.loadMJCF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        return robot_id

    def set_ctrl_mode(self,ctrl_type='tor'):
        pass

    def step_if_not_rtsim(self):
        if not self._rt_sim:
            self.step_sim()

    def step_sim(self):
        pb.stepSimulation(self._uid)

    def ee_pose(self):
        pos,ori = pb.getBasePositionAndOrientation(self._id)
        pos = np.array(pos)
        ori = np.quaternion(ori[3],ori[0],ori[1],ori[2])
        return pos, ori

    def ee_velocity(self):
        lin_vel,ang_vel = pb.getBaseVelocity(self._id)
        lin_vel = np.asarray(lin_vel)
        ang_vel = np.asarray(ang_vel)
        return lin_vel, ang_vel

    def get_link_pose(self, link_id=-3):
        """
        :return: Pose of link (Cartesian positionof center of mass,
                            Cartesian orientation of center of mass in quaternion [x,y,z,w])
                            modified to pose of URDF link frame
        :rtype: [np.ndarray, np.quaternion]

        :param link_id: optional parameter to specify the link id. If not provided,
                        will return pose of end-effector
        :type link_id: int
        """
        link_state = pb.getLinkState(
            self._id, link_id, physicsClientId=self._uid)
        pos = np.asarray(link_state[4])
        ori = np.quaternion(link_state[5][3], link_state[5][0], link_state[5][1],
                            link_state[5][2])  # hamilton convention
        return pos, ori


class Robotiq85GripperFloating():

    def __init__(self,
                 robot_description=floating_description_path,
                 useFixedBase=True,
                 uid=None,
                 gravity=False,
                 realtime_sim=False):
        """
        :param robot_description: path to description file (urdf, .bullet, etc.)
        :param config: optional config file for specifying robot information
        :param uid: optional server id of bullet

        :type robot_description: str
        :type config: dict
        :type uid: int
        """
        self._ready = False
        if uid is None:
            uid = pb.connect(pb.GUI)

        self._uid = uid
        robot_id = self.load_model(robot_description, useFixedBase)
        self._id = robot_id

        if gravity:
            pb.setGravity(0.0, 0.0, -9.8, physicsClientId=self._uid)
        else:
            pb.setGravity(0.0, 0.0, 0., physicsClientId=self._uid)

        self.gravity = gravity

        if realtime_sim:
            pb.setRealTimeSimulation(1, physicsClientId=self._uid)
            pb.setTimeStep(0.01, physicsClientId=self._uid)
        self._rt_sim = realtime_sim

        self.gripper_range = [0, 0.085]

        # by default, set FT sensor at last fixed joint
        # self._ft_joints = [0]
        # self.set_ft_sensor_at(self._ft_joints[0])
        self.__parse_joint_info__()
        self.__post_load__()

        self._all_joints = np.array(range(pb.getNumJoints(self._id, physicsClientId=self._uid)))
        self._movable_joints = np.array(self.controllable_joints)

        self.ee_link = 6
        self.pose_joint = [0, 1, 2, 3, 4, 5]

        self._ready = True
        # import pdb;pdb.set_trace()

    def move_pose(self, cmd):
        # cmd = [x,y,z,r,p,y]
        for i in self.pose_joint:
            pb.setJointMotorControl2(self._id,i,pb.POSITION_CONTROL,targetPosition=cmd[i],force=10000)
        [pb.stepSimulation() for i in range(15)]

    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = pb.createConstraint(self._id, self.mimic_parent_id,
                                   self._id, joint_id,
                                   jointType=pb.JOINT_GEAR,
                                   jointAxis=[1, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            pb.changeConstraint(c, gearRatio=-multiplier, maxForce=10000, erp=1)  # Note: the mysterious `erp` is of EXTREME importance


    def __parse_joint_info__(self):
        numJoints = pb.getNumJoints(self._id)
        jointInfo = namedtuple('jointInfo',['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                                'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = pb.getJointInfo(self._id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != pb.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                pb.setJointMotorControl2(self._id, jointID, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

    def set_ft_sensor_at(self, joint_id, enable=True):
        print("FT sensor at joint", joint_id)
        pb.enableJointForceTorqueSensor(self._id, joint_id, enable, self._uid)


    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        pb.setJointMotorControl2(self._id, self.mimic_parent_id, pb.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.joints[self.mimic_parent_id].maxForce, maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        [pb.stepSimulation() for i in range(10)]


    # def hold_gripper(self,force=600):
    #     for i in [1, 2]:
    #         pb.setJointMotorControl2(self._id, i, pb.TORQUE_CONTROL, force=force)
    #     [pb.stepSimulation() for _ in range(10)]

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def load_model(self,robot_description, useFixedBase):
        extension = robot_description.split('.')[-1]
        if extension == "urdf":
            robot_id = pb.loadURDF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        elif extension == 'sdf':
            robot_id = pb.loadSDF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        elif extension == 'bullet':
            robot_id = pb.loadBullet(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        else:
            robot_id = pb.loadMJCF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid)
        return robot_id

    def set_ctrl_mode(self,ctrl_type='tor'):
        pass

    def step_if_not_rtsim(self):
        if not self._rt_sim:
            self.step_sim()

    def step_sim(self):
        pb.stepSimulation(self._uid)

    def ee_pose(self):
        pos,ori = pb.getBasePositionAndOrientation(self._id)
        pos = np.array(pos)
        ori = np.quaternion(ori[3],ori[0],ori[1],ori[2])
        return pos, ori

    def ee_velocity(self):
        lin_vel,ang_vel = pb.getBaseVelocity(self._id)
        lin_vel = np.asarray(lin_vel)
        ang_vel = np.asarray(ang_vel)
        return lin_vel, ang_vel

    def get_link_pose(self, link_id=-3):
        """
        :return: Pose of link (Cartesian positionof center of mass,
                            Cartesian orientation of center of mass in quaternion [x,y,z,w])
                            modified to pose of URDF link frame
        :rtype: [np.ndarray, np.quaternion]

        :param link_id: optional parameter to specify the link id. If not provided,
                        will return pose of end-effector
        :type link_id: int
        """
        link_state = pb.getLinkState(
            self._id, link_id, physicsClientId=self._uid)
        pos = np.asarray(link_state[4])
        ori = np.quaternion(link_state[5][3], link_state[5][0], link_state[5][1],
                            link_state[5][2])  # hamilton convention
        return pos, ori


class PadGripperFloating():

    def __init__(self,
                 robot_description=floating_pad_gripper_description_path,
                 useFixedBase=True,
                 uid=None,
                 gravity=False,
                 realtime_sim=False,
                 globalScaling=1.0,):
        """
        :param robot_description: path to description file (urdf, .bullet, etc.)
        :param config: optional config file for specifying robot information
        :param uid: optional server id of bullet

        :type robot_description: str
        :type config: dict
        :type uid: int
        """
        self._ready = False
        if uid is None:
            uid = pb.connect(pb.GUI)

        self._uid = uid
        robot_id = self.load_model(robot_description, useFixedBase, globalScaling=globalScaling)
        self._id = robot_id

        if gravity:
            pb.setGravity(0.0, 0.0, -9.8, physicsClientId=self._uid)
        else:
            pb.setGravity(0.0, 0.0, 0., physicsClientId=self._uid)

        self.gravity = gravity

        if realtime_sim:
            pb.setRealTimeSimulation(1, physicsClientId=self._uid)
            pb.setTimeStep(0.01, physicsClientId=self._uid)
        self._rt_sim = realtime_sim

        self.gripper_range = [0, 0.1 * globalScaling]

        # by default, set FT sensor at last fixed joint
        # self._ft_joints = [0]
        # self.set_ft_sensor_at(self._ft_joints[0])
        self.__parse_joint_info__()
        # self.__post_load__()

        self._all_joints = np.array(range(pb.getNumJoints(self._id, physicsClientId=self._uid)))
        self._movable_joints = np.array(self.controllable_joints)

        self.ee_link = 6
        self.pose_joint = [0, 1, 2, 3, 4, 5]

        self._ready = True
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == 'left_finger_pad_joint'][0]
        # import pdb;pdb.set_trace()

    def move_pose(self, cmd):
        # cmd = [x,y,z,r,p,y] rot(y)@rot(p)@rot(r)
        for i in self.pose_joint:
            pb.setJointMotorControl2(self._id,i,pb.POSITION_CONTROL,targetPosition=cmd[i],force=10000)
        # [pb.stepSimulation() for _ in range(15)]

    def gripper_pose(self,return_list=True):
        cmd = []
        for i in self.pose_joint:
            jointPosition, _, _ ,_ = pb.getJointState(self._id,i)
            cmd.append(jointPosition)
        pose = np.identity(4)
        pose[:3, :3] = Rot.from_euler('ZYX', cmd[3:], degrees=False).as_matrix()
        pose[:3, 3] = np.array(cmd[:3])
        if return_list:
            return cmd
        else:
            return pose



    def __post_load__(self):
        # To control the gripper
        mimic_parent_name = 'left_finger_pad_joint'
        mimic_children_names = {'right_finger_pad_joint': 1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = pb.createConstraint(self._id, self.mimic_parent_id,
                                   self._id, joint_id,
                                   jointType=pb.JOINT_GEAR,
                                   jointAxis=[1, 0, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            pb.changeConstraint(c, gearRatio=multiplier, maxForce=10000, erp=1)  # Note: the mysterious `erp` is of EXTREME importance


    def __parse_joint_info__(self):
        numJoints = pb.getNumJoints(self._id)
        jointInfo = namedtuple('jointInfo',['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                                'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = pb.getJointInfo(self._id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != pb.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                pb.setJointMotorControl2(self._id, jointID, pb.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

    def set_ft_sensor_at(self, joint_id, enable=True):
        print("FT sensor at joint", joint_id)
        pb.enableJointForceTorqueSensor(self._id, joint_id, enable, self._uid)


    def move_gripper(self, open_length,force=600,wait_done=True):
        # open_length = np.clip(open_length, *self.gripper_range)
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]+0.01
        # pb.setJointMotorControl2(self._id, self.mimic_parent_id, pb.POSITION_CONTROL, targetPosition=open_length/2.,
        #                          force=self.joints[self.mimic_parent_id].maxForce,
        #                          maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        for i in [self.mimic_parent_id, self.mimic_parent_id+1]:
            pb.setJointMotorControl2(self._id, i, pb.POSITION_CONTROL, open_length/2., force=force)
        [pb.stepSimulation() for _ in range(1000)]
        open_done = False
        steps = 0
        while not open_done and wait_done:
            pb.stepSimulation()
            steps += 1
            joint1 = pb.getJointState(self._id, self.mimic_parent_id)[0]
            joint2 = pb.getJointState(self._id, self.mimic_parent_id+1)[0]
            if abs(joint1-open_length/2)<0.0001 and abs(joint2-open_length/2.)<0.0001:
                open_done = True
            if steps > 1000 * 3:
                break


    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self,wait_done=True):
        self.move_gripper(self.gripper_range[0],wait_done=wait_done)

    def load_model(self,robot_description, useFixedBase, globalScaling):
        extension = robot_description.split('.')[-1]
        if extension == "urdf":
            robot_id = pb.loadURDF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid, globalScaling=globalScaling)
        elif extension == 'sdf':
            robot_id = pb.loadSDF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid, globalScaling=globalScaling)
        elif extension == 'bullet':
            robot_id = pb.loadBullet(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid, globalScaling=globalScaling)
        else:
            robot_id = pb.loadMJCF(
                robot_description, useFixedBase=useFixedBase, physicsClientId=self._uid, globalScaling=globalScaling)
        return robot_id

    def set_ctrl_mode(self,ctrl_type='tor'):
        pass

    def step_if_not_rtsim(self):
        if not self._rt_sim:
            self.step_sim()

    def step_sim(self):
        pb.stepSimulation(self._uid)

    def ee_pose(self):
        pos,ori = pb.getBasePositionAndOrientation(self._id)
        pos = np.array(pos)
        ori = np.quaternion(ori[3],ori[0],ori[1],ori[2])
        return pos, ori

    def ee_velocity(self):
        lin_vel,ang_vel = pb.getBaseVelocity(self._id)
        lin_vel = np.asarray(lin_vel)
        ang_vel = np.asarray(ang_vel)
        return lin_vel, ang_vel

    def get_link_pose(self, link_id=-3):
        """
        :return: Pose of link (Cartesian positionof center of mass,
                            Cartesian orientation of center of mass in quaternion [x,y,z,w])
                            modified to pose of URDF link frame
        :rtype: [np.ndarray, np.quaternion]

        :param link_id: optional parameter to specify the link id. If not provided,
                        will return pose of end-effector
        :type link_id: int
        """
        link_state = pb.getLinkState(
            self._id, link_id, physicsClientId=self._uid)
        pos = np.asarray(link_state[4])
        ori = np.quaternion(link_state[5][3], link_state[5][0], link_state[5][1],
                            link_state[5][2])  # hamilton convention
        return pos, ori

    def __del__(self):
        pb.removeBody(self._id)
