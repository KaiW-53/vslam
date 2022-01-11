import numpy as np
import matplotlib.pyplot as plt

traj1 = np.loadtxt("./build/original.txt")
traj1 = traj1[np.argsort(traj1[:, 0])]

traj2 = np.loadtxt("./build/traj_between_opti5.txt")
traj2 = traj2[np.argsort(traj2[:, 0])]

#traj3 = np.loadtxt("./build/all_opti.txt")
#traj3 = traj3[np.argsort(traj3[:, 0])]

gt = np.loadtxt("../data/data_odometry_poses/dataset/poses/00.txt")
plt.plot(traj1[:, 4], traj1[:, 12], color = 'r')
plt.plot(traj2[:, 1], traj2[:, 2], color = 'g')
plt.plot(gt[:, 3], gt[:, 11], color = 'b')
#plt.plot(traj3[:, 1], traj3[:, 2], color = 'b')
plt.show()