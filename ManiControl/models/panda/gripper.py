import os
import numpy as np
import pybullet as pb
import quaternion

description_path = os.path.dirname(
    os.path.abspath(__file__)) + "/objects/gripper_panda.urdf"

floating_description_path = os.path.dirname(
    os.path.abspath(__file__)) + "/objects/gripper_panda_combined.urdf"

class PandaGripper():

    def __init__(self,
                 robot_description=description_path,
                 useFixedBase=False,
                 uid=None,
                 gravity=False,
                 realtime_sim=False,
                 globalScaling=1.0):
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
        # pb.resetSimulation(physicsClientId=self._uid)
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

        # by default, set FT sensor at last fixed joint
        self._ft_joints = [0]
        self.set_ft_sensor_at(self._ft_joints[0])

        self._all_joints = np.array(
            range(pb.getNumJoints(self._id, physicsClientId=self._uid)))
        self._movable_joints = self.get_movable_joints()

        self._ready = True

        self.gripper_range = [0, 0.04 * globalScaling]
        c = pb.createConstraint(self._id,
                               1,
                               self._id,
                               2,
                               jointType=pb.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=500)
        self.ft_direction = None

    def set_ft_sensor_at(self, joint_id, enable=True):
        pb.enableJointForceTorqueSensor(self._id, joint_id, enable, self._uid)

    def get_movable_joints(self):
        """
        :return: Ids of all movable joints.
        :rtype: np.ndarray (shape: (self._nu,))

        """
        movable_joints = []
        for i in self._all_joints:
            joint_info = pb.getJointInfo(
                self._id, i, physicsClientId=self._uid)
            q_index = joint_info[3]
            if q_index > -1:
                movable_joints.append(i)
        return np.array(movable_joints)

    def move_gripper(self, open_length, force=600):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in [1, 2]:
            pb.setJointMotorControl2(self._id, i, pb.POSITION_CONTROL, open_length, force=force)

        # pb.setJointMotorControl2(self._id, 9, pb.POSITION_CONTROL, open_length, force=20)
        [pb.stepSimulation() for _ in range(10)]

    def hold_gripper(self, force=600):
        for i in [1, 2]:
            pb.setJointMotorControl2(self._id, i, pb.TORQUE_CONTROL, force=force)
        [pb.stepSimulation() for _ in range(10)]

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def load_model(self,robot_description, useFixedBase, globalScaling=1.0):
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

    def exec_torque_cmd(self, cmd):
        pos, ori = pb.getBasePositionAndOrientation(self._id)
        fx, fy, fz, tx, ty, tz = cmd
        pb.applyExternalForce(self._id, -1, [fx, fy, fz], pos, pb.WORLD_FRAME)
        pb.applyExternalTorque(self._id, -1, [tx, ty, tz], pb.WORLD_FRAME)
        # kwargs = dict()
        # kwargs['lineWidth'] = 10.
        # kwargs['lineColorRGB'] = [1, 0, 0]
        # if self.ft_direction is not None:
        #     kwargs['replaceItemUniqueId'] = self.ft_direction
        # self.ft_direction = pb.addUserDebugLine(pos, np.array(pos)+np.array([fx, fy, fz]), **kwargs)


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

    def get_joint_state(self, joint_id=None):
        """
        :return: joint positions, velocity, reaction forces, joint efforts as given from
                bullet physics
        :rtype: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if joint_id is None:
            joint_angles = []
            joint_velocities = []
            joint_reaction_forces = []
            joint_efforts = []

            for idx in self._movable_joints:
                joint_state = pb.getJointState(
                    self._id, idx, physicsClientId=self._uid)

                joint_angles.append(joint_state[0])

                joint_velocities.append(joint_state[1])

                joint_reaction_forces.append(joint_state[2])

                joint_efforts.append(joint_state[3])

            return np.array(joint_angles), np.array(joint_velocities), np.array(joint_reaction_forces), np.array(
                joint_efforts)

        else:
            jnt_state = pb.getJointState(
                self._id, joint_id, physicsClientId=self._uid)

            jnt_poss = jnt_state[0]

            jnt_vels = jnt_state[1]

            jnt_reaction_forces = jnt_state[2]

            jnt_applied_torques = jnt_state[3]

            return jnt_poss, jnt_vels, np.array(jnt_reaction_forces), jnt_applied_torques

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

    def get_ee_wrench(self, local=False, verbose=False):
        '''
        :param local: if True, computes reaction forces in local sensor frame, else in base frame of robot
        :type local: bool
        :return: End effector forces and torques. Returns [fx, fy, fz, tx, ty, tz]
        :rtype: np.ndarray
        '''

        _, _, jnt_reaction_force, _ = self.get_joint_state(self._ft_joints[-1])
        if not local:
            jnt_reaction_force = np.asarray(jnt_reaction_force)  # below add -
            ee_pos, ee_ori = self.get_link_pose(self._ft_joints[-1])
            rot_mat = quaternion.as_rotation_matrix(ee_ori)
            f = np.dot(rot_mat, np.asarray([jnt_reaction_force[0], jnt_reaction_force[1], jnt_reaction_force[2]]))
            t = np.dot(rot_mat, np.asarray([jnt_reaction_force[0+3], jnt_reaction_force[1+3], jnt_reaction_force[2+3]]))
            jnt_reaction_force = np.append(f,t).flatten()

        return jnt_reaction_force


class PandaGripperFloating():
    def __init__(self,
                 robot_description=floating_description_path,
                 useFixedBase=False,
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

        # by default, set FT sensor at last fixed joint
        self._ft_joints = [6]
        self.set_ft_sensor_at(self._ft_joints[0])

        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self._all_joints = np.array(
            range(pb.getNumJoints(self._id, physicsClientId=self._uid)))
        # [0, 1, 2, 3, 4, 5, 7, 8]
        self._movable_joints = self.get_movable_joints()

        self._ready = True

        self.ee_link = 6
        self.gripper_range = [0, 0.04]
        self.finger_joint = [7,8]
        self.pose_joint = [0,1,2,3,4,5]
        c = pb.createConstraint(self._id,
                               self.finger_joint[0],
                               self._id,
                               self.finger_joint[1],
                               jointType=pb.JOINT_GEAR,
                               jointAxis=[0, 1, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=500)
        self.ft_direction = None

    def set_ft_sensor_at(self, joint_id, enable=True):
        print("FT sensor at joint", joint_id)
        pb.enableJointForceTorqueSensor(self._id, joint_id, enable, self._uid)

    def get_movable_joints(self):
        """
        :return: Ids of all movable joints.
        :rtype: np.ndarray (shape: (self._nu,))

        """
        movable_joints = []
        for i in self._all_joints:
            joint_info = pb.getJointInfo(
                self._id, i, physicsClientId=self._uid)
            q_index = joint_info[3]
            if q_index > -1:
                movable_joints.append(i)
        return np.array(movable_joints)

    def move_gripper(self, open_length, force=600):
        assert self.gripper_range[0] <= open_length <= self.gripper_range[1]
        for i in self.finger_joint:
            pb.setJointMotorControl2(self._id, i, pb.POSITION_CONTROL, open_length, force=force)

        # pb.setJointMotorControl2(self._id, 9, pb.POSITION_CONTROL, open_length, force=20)
        [pb.stepSimulation() for _ in range(10)]

    def hold_gripper(self,force=600):
        for i in self.finger_joint:
            pb.setJointMotorControl2(self._id, i, pb.TORQUE_CONTROL, force=force)
        [pb.stepSimulation() for _ in range(10)]

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

    def exec_torque_cmd(self,cmd):
        pos, ori = pb.getBasePositionAndOrientation(self._id)
        fx,fy,fz,tx,ty,tz = cmd
        pb.applyExternalForce(self._id,-1,[fx,fy,fz],pos,pb.WORLD_FRAME)
        pb.applyExternalTorque(self._id,-1,[tx,ty,tz],pb.WORLD_FRAME)
        kwargs = dict()
        kwargs['lineWidth'] = 10.
        kwargs['lineColorRGB'] = [1, 0, 0]
        if self.ft_direction is not None:
            kwargs['replaceItemUniqueId'] = self.ft_direction
        self.ft_direction = pb.addUserDebugLine(pos, np.array(pos)+np.array([fx,fy,fz])*0.1,**kwargs)


    def set_ctrl_mode(self,ctrl_type='tor'):
        pass

    def step_if_not_rtsim(self):
        if not self._rt_sim:
            self.step_sim()

    def step_sim(self):
        pb.stepSimulation(self._uid)

    def ee_pose(self):
        link_state = pb.getLinkState(self._id,self.ee_link)
        pos, ori = link_state[4], link_state[5]
        # pb.getBasePositionAndOrientation(self._id)
        R_mat = np.array(pb.getMatrixFromQuaternion(ori)).reshape((3,3))
        pose = np.identity(4)
        pose[:3,:3] = R_mat
        pose[:3,3] = np.array(pos)
        return pose

    def ee_velocity(self):
        link_state = pb.getLinkState(self._id, self.ee_link,computeLinkVelocity=True)
        lin_vel,ang_vel = link_state[6:8]
        # lin_vel,ang_vel = pb.getBaseVelocity(self._id)
        lin_vel = np.asarray(lin_vel)
        ang_vel = np.asarray(ang_vel)
        return lin_vel, ang_vel

    def get_joint_state(self, joint_id=None):
        """
        :return: joint positions, velocity, reaction forces, joint efforts as given from
                bullet physics
        :rtype: [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """
        if joint_id is None:
            joint_angles = []
            joint_velocities = []
            joint_reaction_forces = []
            joint_efforts = []

            for idx in self._movable_joints:
                joint_state = pb.getJointState(
                    self._id, idx, physicsClientId=self._uid)

                joint_angles.append(joint_state[0])

                joint_velocities.append(joint_state[1])

                joint_reaction_forces.append(joint_state[2])

                joint_efforts.append(joint_state[3])

            return np.array(joint_angles), np.array(joint_velocities), np.array(joint_reaction_forces), np.array(
                joint_efforts)

        else:
            jnt_state = pb.getJointState(
                self._id, joint_id, physicsClientId=self._uid)

            jnt_poss = jnt_state[0]

            jnt_vels = jnt_state[1]

            jnt_reaction_forces = jnt_state[2]

            jnt_applied_torques = jnt_state[3]

            return jnt_poss, jnt_vels, np.array(jnt_reaction_forces), jnt_applied_torques

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

    def get_ee_wrench(self, local=False):
        '''
        :param local: if True, computes reaction forces in local sensor frame, else in base frame of robot
        :type local: bool
        :return: End effector forces and torques. Returns [fx, fy, fz, tx, ty, tz]
        :rtype: np.ndarray
        '''

        _, _, jnt_reaction_force, _ = self.get_joint_state(self._ft_joints[-1])
        if not local:
            jnt_reaction_force = np.asarray(jnt_reaction_force)  # below add -
            ee_pos, ee_ori = self.get_link_pose(self._ft_joints[-1])
            rot_mat = quaternion.as_rotation_matrix(ee_ori)
            f = np.dot(rot_mat, np.asarray([jnt_reaction_force[0], jnt_reaction_force[1], jnt_reaction_force[2]]))
            t = np.dot(rot_mat, np.asarray([jnt_reaction_force[0+3], jnt_reaction_force[1+3], jnt_reaction_force[2+3]]))
            jnt_reaction_force = np.append(f,t).flatten()

        return jnt_reaction_force

    def move_pose(self,cmd):
        # cmd = [x,y,z,r,p,y]
        for i in self.pose_joint:
            pb.setJointMotorControl2(self._id,i,pb.POSITION_CONTROL,targetPosition=cmd[i])


if __name__ == '__main__':

    panda = PandaGripperFloating(gravity=False)
    # panda.open_gripper()
    panda.move_pose([0.2,0.3,0.4,0.1,0.2,0.3])
    [pb.stepSimulation() for _ in range(100)]
    # import pdb;pdb.set_trace()
    while 1:
        pb.stepSimulation()
    # pass
