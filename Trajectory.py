import matplotlib
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from quad_model import Model
from pyquaternion import Quaternion


class Trajectory:
    def __init__(self):
        self.poses = []

    def run_test(self, agent):
        # Run a test on trained agent to generate data for plotting
        task = agent.task
        state = agent.reset_episode()
        total_reward = 0
        self.poses.append(task.sim.pose)
        while True:
            action = agent.test_act(state)
            next_state, reward, done = task.step(action)
            self.poses.append(task.sim.pose)
            total_reward += reward
            state = next_state
            if done:
                print("flight time: {:5.3f} seconds, total reward: {:7.3f}".format(task.sim.time, total_reward))
                np.set_printoptions(formatter={'float': '{:7.3f}'.format})
                print("final position: {}".format(task.sim.pose[:3]))
                print("finished running test; available for plotting.")
                break

    def reset(self):
        self.poses = []

    def plot_trajectory(self, show_orientations=False, every_n=1, initial_position=[0, 0, 150], end_time=5.2):
        positions = np.array([pose[:3] for pose in self.poses])
        orientations = np.array([pose[3:] for pose in self.poses])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D trajectory plot")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # fig.show()
        fig.canvas.draw()

        plt.ion()

        x_range = np.amax(np.abs(positions[:, 0]))-np.amin(positions[:, 0])
        y_range = np.amax(np.abs(positions[:, 1]))-np.amin(positions[:, 1])
        z_range = np.amax(np.abs(positions[:, 2]))-np.amin(positions[:, 2])
        n = max([x_range, y_range, z_range]) * 1.2
        x_initial, y_initial, z_initial = initial_position
        scale = 1.
        ax.set_xlim((x_initial-n*scale, x_initial+n*scale))
        ax.set_ylim((y_initial-n*scale, y_initial+n*scale))
        ax.set_zlim((z_initial-n*scale, z_initial+n*scale))
        ax.plot(xs=positions[:, 0], ys=positions[:, 1], zs=positions[:, 2],
                c='k', marker='.', markersize=5)

        if show_orientations:
            # model = Model(ax, length=n / 2)
            for i, euler, pos in zip(range(len(orientations)), orientations, positions):
                if i%every_n == 0 or i == len(orientations)-1: # Set condition for plotting this orientation
                    quaternion = self.euler2quat(euler)
                    model = Model(ax, length=[n/2]*3)
                    model.set_pos(pos)
                    model.rotate(quaternion)
                    fig.canvas.draw()
                    # model.redraw()
                    # fig.canvas.update()
                    fig.canvas.flush_events()
                    # plt.pause(0.001)
                    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("2D plot")
        time=np.linspace(0, end_time, len(positions))
        ax.plot(time, positions[:, 0], label='x')
        ax.plot(time, positions[:, 1], label='y')
        ax.plot(time, positions[:, 2], label='z')
        ax.legend()

    def euler2quat(self, euler_angles):
        roll, pitch, yaw = euler_angles

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        w = cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp
        return Quaternion(w, x, y, z)
