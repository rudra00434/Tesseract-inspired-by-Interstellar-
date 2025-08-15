import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.animation import FuncAnimation
np.random.seed(42)
stars = np.random.uniform(-5, 5, (500, 3)) 

vertices4 = np.array(list(product([-1, 1], repeat=4)), dtype=float)


def hamming_distance(a, b):
    return int(np.sum(a != b))

edges = [(i, j) for i in range(len(vertices4)) for j in range(i+1, len(vertices4))
         if hamming_distance(vertices4[i], vertices4[j]) == 1]


def rotation_4d(theta_xy, theta_zw, theta_xw, theta_yz):
    def R_plane(n, i, j, theta):
        R = np.eye(n)
        c, s = np.cos(theta), np.sin(theta)
        R[i, i], R[j, j] = c, c
        R[i, j], R[j, i] = -s, s
        return R

    R = np.eye(4)
    R = R_plane(4, 0, 1, theta_xy) @ R
    R = R_plane(4, 2, 3, theta_zw) @ R
    R = R_plane(4, 0, 3, theta_xw) @ R
    R = R_plane(4, 1, 2, theta_yz) @ R
    return R


def project_to_3d(vertices4):
    d = 3.0
    w = vertices4[:, 3:4]
    scale = 1.0 / (d - w)
    return vertices4[:, :3] * scale


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()

max_range = 2.5
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)


def update(frame):
    ax.cla()
    ax.set_axis_off()
    ax.set_facecolor('black')
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    ax.set_title("Rotating Tesseract (4D -> 3D Projection)", pad=12)
    ax.scatter(stars[:, 0], stars[:, 1], stars[:, 2], c='white', s=0.5, alpha=0.5)
    
    theta = frame * 0.05
    R4 = rotation_4d(theta, theta*0.9, theta*0.7, theta*1.1)
    rotated4 = vertices4 @ R4.T
    proj3 = project_to_3d(rotated4)
    
    
    for i, j in edges:
        ax.plot([proj3[i, 0], proj3[j, 0]],
                [proj3[i, 1], proj3[j, 1]],
                [proj3[i, 2], proj3[j, 2]], 
                color='#00ffff', alpha=0.6, lw=2)
    
    
    ax.scatter(proj3[:, 0], proj3[:, 1], proj3[:, 2], s=30, c='red')


anim = FuncAnimation(fig, update, frames=100, interval=80)


out_gif = r"C:\Users\tatai\OneDrive\Desktop\tesseract_rotation.gif"
anim.save(out_gif, writer='pillow', fps=25)
print(f"Saved animation to {out_gif}")

plt.close(fig)

out_gif              