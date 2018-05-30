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

    def plot_trajectory(self, show_orientations=False, every_n=1):
        positions = np.array([pose[:3] for pose in self.poses])
        orientations = np.array([pose[3:] for pose in self.poses])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # fig.show()
        fig.canvas.draw()

        plt.ion()

        n = np.amax(np.abs(positions)) * 1.2
        ax.set_xlim((-n, n))
        ax.set_ylim((-n, n))
        ax.set_zlim((-n, n))
        ax.plot(xs=positions[:, 0], ys=positions[:, 1], zs=positions[:, 2],
                c='k', marker='.', markersize=5)

        if show_orientations:
            # model = Model(ax, length=n / 2)
            for i, euler, pos in zip(range(len(orientations)), orientations, positions):
                if i%every_n == 0 or i == len(orientations)-1: # Set condition for plotting this orientation
                    quaternion = self.euler2quat(euler)
                    model = Model(ax, length=n/2)
                    model.set_pos(pos)
                    model.rotate(quaternion)
                    fig.canvas.draw()
                    # model.redraw()
                    # fig.canvas.update()
                    fig.canvas.flush_events()
                    # plt.pause(0.001)

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
