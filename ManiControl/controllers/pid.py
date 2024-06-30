from typing import Tuple
import threading
import numpy as np
import quaternion
import time
from collections import deque
import pybullet as pb

def quatdiff_in_euler(quat_curr:np.ndarray, quat_des:np.ndarray) -> np.ndarray:
    curr_mat = quaternion.as_rotation_matrix(quat_curr)
    des_mat = quaternion.as_rotation_matrix(quat_des)

    rel_mat = des_mat.T.dot(curr_mat)

    rel_quat = quaternion.from_rotation_matrix(rel_mat)

    vec = quaternion.as_float_array(rel_quat)[1:]

    if rel_quat.w < 0.0:
        vec = -vec

    return -des_mat.dot(vec)


ImpConfig = {
    'KP_pos': [1000., 1000., 1000.], 
    'KI_pos': [0., 0., 0.], 
    'KD_pos': [65., 65., 65.], 
    'KP_ori': [300., 300., 300.], 
    'KI_ori': [0., 0., 0.], 
    'KD_ori': [0.01, 0.01, 0.01], 
    'ft_thresh': [100., 100., 100., 100., 100., 100.], 
    'initial_err': [200., 200.], 
    'error_thresh': [0.001, 0.005], 
    'history_len': 300
}


class PIDController(object):
    def __init__(self, robot, config:dict=ImpConfig, break_condition=None) -> None:
        self._robot = robot # robot sim should not be in real-time. Step simulation will be called by controller.
        
        self._KP_pos = np.diag(config['KP_pos'])
        self._KI_pos = np.diag(config['KI_pos'])
        self._KD_pos = np.diag(config['KD_pos'])
        self._KP_ori = np.diag(config['KP_ori'])
        self._KI_ori = np.diag(config['KI_ori'])
        self._KD_ori = np.diag(config['KD_ori'])
        self._ft_thresh = np.array(config['ft_thresh'])
        self._initial_err = np.array(config['initial_err'])
        self._error_thresh = np.array(config['error_thresh'])

        self._run_ctrl = False

        if break_condition is not None and callable(break_condition):
            self._break_condition = break_condition
        else:
            self._break_condition = lambda: False

        self._ctrl_thread = None
        self._mutex = threading.Lock()

        self._sim_timestep = pb.getPhysicsEngineParameters()['fixedTimeStep']
        self._sim_time = 0.0

        if 'rate' not in config:
            self._ctrl_rate = 1./self._sim_timestep
        else:
            self._ctrl_rate = float(config['rate'])
        
        maxlen = config['history_len']
        self.err_deque = deque([], maxlen=maxlen)       # (6,)s
        self.ft_deque = deque([], maxlen=maxlen)        # (6,)s
    
    def update_goal(self, goal_pos:np.ndarray, goal_ori:np.ndarray) -> None:
        """
        Should update the values for self._goal_pos and self._goal_ori at least.
        """
        self._mutex.acquire()
        self._goal_pos = np.copy(goal_pos).reshape([3,1])
        self._goal_ori = np.copy(goal_ori)
        self._mutex.release()

    def _compute_cmd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Should compute the joint torques that are to be applied at every sim step.
        """
        # current observe
        curr_pos, curr_ori = self._robot.ee_pose()
        curr_vel, curr_omg = self._robot.ee_velocity()

        # current error
        delta_pos = self._goal_pos - curr_pos.reshape([3, 1])
        delta_ori = quatdiff_in_euler(curr_ori, self._goal_ori).reshape([3, 1])
        
        self.err_deque.append(np.vstack([delta_pos, delta_ori]).flatten())
        sum_pos = np.sum(np.array(self.err_deque)[:, :3], axis=0).reshape([3, 1])
        sum_ori = np.sum(np.array(self.err_deque)[:, 3:], axis=0).reshape([3, 1])
        error_value = np.asarray([np.linalg.norm(delta_pos), np.linalg.norm(delta_ori)])

        # PID control
        ft = np.vstack([self._KP_pos.dot(delta_pos), self._KP_ori.dot(delta_ori)]) + \
             np.vstack([self._KI_pos.dot(sum_pos), self._KI_ori.dot(sum_ori)]) - \
             np.vstack([self._KD_pos.dot(curr_vel.reshape([3, 1])),
                       self._KD_ori.dot(curr_omg.reshape([3, 1]))])
        ft = np.clip(ft.flatten(), -self._ft_thresh, self._ft_thresh)

        self.ft_deque.append(ft)

        return (ft, error_value)    # (6,) (2,)

    def _control_thread(self):
        """
        Apply the torque command computed in _compute_cmd until any of the 
        break conditions are met.
        """
        while self._run_ctrl and not self._break_condition():
            error = self._initial_err
            while np.any(error > self._error_thresh):
                if not self._run_ctrl or self._break_condition():
                    break
                now = time.time()
                
                self._mutex.acquire()
                tau, error = self._compute_cmd()
                
                # command robot using the computed joint torques
                self._robot.exec_torque_cmd(tau)

                self._robot.step_if_not_rtsim()
                self._sim_time += self._sim_timestep
                self._mutex.release()

                elapsed = time.time() - now
                sleep_time = (1./self._ctrl_rate) - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    def _initialise_goal(self) -> None:
        """
        Should initialise _goal_pos, _goal_ori, etc. for controller to start the loop.
        Ideally these should be the current value of the robot's end-effector.
        """
        self.update_goal(self._robot.ee_pose()[0],self._robot.ee_pose()[1])

    def start_controller_thread(self) -> None:
        self._initialise_goal()
        self._run_ctrl = True
        if self._ctrl_thread is None or not self._ctrl_thread.is_alive():
            self._ctrl_thread = threading.Thread(target=self._control_thread)
        self._ctrl_thread.start()

    def stop_controller_thread(self) -> None:
        self._run_ctrl = False
        if self._ctrl_thread is not None and self._ctrl_thread.is_alive():
            self._ctrl_thread.join()

    def __del__(self) -> None:
        self.stop_controller_thread()
