import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from functools import partial


fig, ax = plt.subplots(figsize=(6,7))

#                        head          lhand       rhand      lfoot        rfoot
# ef_points = np.array([[0.0, 2.0], [ 2.0, 1.0], [-2.0, 1.0], [ 1.0, -2.0], [-1.0, -2.0]]) # ik points
# ef_indices = np.array([2, 4, 6, 8, 11])

# (0, 1), (0, 2), (-1, -1), (1, 1)

anchors = np.array([
    [0, 0], [0, 3], [-2, -2], [2, -2]
])

cur_point = np.array(
    [[0.0, 1.0]]
)

keyposes = np.array([
[0.0] *13,
[0.014,0.042,-0.111,-0.193,1.354,0.071,-1.396,-0.014,0.000,0.000,-0.014,0.000,0.000],
[-0.237, 0.183, 0.108, -0.178, 0.465, -0.044, -1.141, 0.237, 0.000, 0.000, -0.657, 1.175, 0.000],
[0.158, -0.334, 0.341, 0.203, 2.279, 0.026, 0.806, 0.649, -0.639, 0.000, -0.158, -0.000, 0.000],
])
# -----
# ('3')
# __|__
#   |
#  / \
# _| |_
#            0 root,      1 spine,     2 head,     3 larm,   4 lhand,      5 rarm,       6 rhand      7 lthig    8 lleg     9 lfoot      10 rthig      11 rleg      12 rfoot
skeleton = [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0],  [1.0, -1.0], [0.0, -1.0], [0.5, 0.0], [-1.0, -1.0], [0.0, -1.0], [-0.5, 0.0]]
skeleton = np.array(skeleton)
parents = [-1, 0, 1, 1, 3, 1, 5, 0, 7, 8, 0, 10, 11]
joint_rotation = np.zeros_like(skeleton[:, 0])
ax.plot(skeleton)
sc = ax.scatter(anchors[:, 0], anchors[:, 1], s=200)
sc2 = ax.scatter(cur_point[:, 0], cur_point[:, 1], s=200)

# rotmat: 관절의 회전 각도를 받아 회전 행렬을 반환
def rotmat(rot):
    return np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

# fk: 각 관절의 위치와 회전 행렬을 반환
def fk(rot): # rot -> pos, rot
    rot = list(map(rotmat, rot))
    global_rots = []
    poss = []
    for i, p in enumerate(parents):
        if p == -1:
            global_rots.append(rot[i])
            poss.append(np.zeros_like(skeleton[0]))
            continue
        pos = poss[p] + global_rots[p]@rot[i]@skeleton[i]
        poss.append(pos)
        global_rots.append(global_rots[p]@rot[i])

    return np.array(poss), np.array(global_rots)

# global_fk: 각 관절의 위치를 반환
def global_fk(rot): # rot -> pos
    poss = []
    for i, p in enumerate(parents):
        if p == -1:
            poss.append(np.zeros_like(skeleton[0]))
            continue
        pos = poss[p] + rot[i]@skeleton[i]
        poss.append(pos)

    return np.array(poss)

# rbf: Radial Basis Function
def rbf(x):
    return np.abs(x).sum()

# interp_func: 보간 함수
def interp_func(c, x):
    retval = 0.0
    for i in range(len(anchors)):
        retval += c[i] * rbf(x - anchors[i])
    return retval + c[-3] + c[-2] * x[0] + c[-1] * x[1]

def solve_spatial_interp(anchors, keyposes):
    n = len(anchors) + 3
    grots = list(map(lambda x: fk(x)[1], keyposes))

    A = np.zeros([n, n])

    # <--- code here (Building matrix A) --->
    for i in range(len(anchors)):
        for j in range(n):
            if j < len(anchors):
                A[i, j] = rbf(anchors[i] - anchors[j])
            if j == len(anchors):
                A[i, j] = 1
            if j == len(anchors) + 1:
                A[i, j] = anchors[i, 0]
            if j == len(anchors) + 2:
                A[i, j] = anchors[i, 1]

    for j in range(n):
        if j < len(anchors):
            A[len(anchors), j] = 1
            A[len(anchors) + 1, j] = anchors[j, 0]
            A[len(anchors) + 2, j] = anchors[j, 1]
    # end

    grots = np.array(grots)
    grots = grots.reshape(len(grots), -1)
    interp_funcs = []
    for i in range(grots.shape[-1]):
        b = np.zeros(n)
        b[:len(anchors)] = grots[:, i]
        interp_funcs.append(partial(interp_func, np.linalg.solve(A, b)))

    return interp_funcs

def fk_n_draw(rot):
    global sc
    global sc2
    ax.cla()
    ax.grid()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    pos, _ = fk(rot)
    for i, p in enumerate(parents):
        if p==-1:
            continue
        ax.plot(pos[[i, p], 0], pos[[i, p], 1], c='k', lw=3)
    ax.text(pos[2, :1], pos[2, 1:], s="('-')", fontsize=30, horizontalalignment='center')
    sc = ax.scatter(anchors[:, 0], anchors[:, 1], s=200)
    sc2 = ax.scatter(cur_point[:, 0], cur_point[:, 1], s=200)

def global_fk_n_draw(rot):
    global sc
    global sc2
    ax.cla()
    ax.grid()
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    pos = global_fk(rot)
    for i, p in enumerate(parents):
        if p==-1:
            continue
        ax.plot(pos[[i, p], 0], pos[[i, p], 1], c='k', lw=3)
    ax.text(pos[2, :1], pos[2, 1:], s="('-')", fontsize=30, horizontalalignment='center')
    sc = ax.scatter(anchors[:, 0], anchors[:, 1], s=200)
    sc2 = ax.scatter(cur_point[:, 0], cur_point[:, 1], s=200)

fk_n_draw([0]*13)

interp_funcs = solve_spatial_interp(anchors, keyposes)

def interp(x):
    rot_vals = []
    for f in interp_funcs:
        rot_vals.append(f(x))

    rot_vals = np.array(rot_vals)
    rot_vals = rot_vals.reshape(-1, 2, 2)

    max_iter = 10
    threshold = 0.000001

    for i in range(len(rot_vals)):
        M = rot_vals[i]
        # 2D 벡터를 3D로 확장 (z=0 추가)
        x0 = np.array([M[0, 0], M[1, 0], 0])
        y0 = np.array([M[0, 1], M[1, 1], 0])

        # z축은 x와 y의 외적으로 계산
        z0 = np.cross(x0, y0)

        x0 = x0 / np.linalg.norm(x0)
        y0 = y0 / np.linalg.norm(y0)
        z0 = z0 / np.linalg.norm(z0)

        for _ in range(max_iter):
            u0 = np.cross(y0, z0)
            v0 = np.cross(z0, x0)
            w0 = np.cross(x0, y0)

            u0 = u0 / np.linalg.norm(u0)
            v0 = v0 / np.linalg.norm(v0)
            w0 = w0 / np.linalg.norm(w0)

            x1 = (x0 + u0) / 2
            y1 = (y0 + v0) / 2
            z1 = (z0 + w0) /2

            x1 = x1 / np.linalg.norm(x1)
            y1 = y1 / np.linalg.norm(y1)
            z1 = z1 / np.linalg.norm(z1)

            residual = np.dot(x1[:2], y1[:2])**2
            if residual < threshold:
                break

            x0, y0, z0 = x1, y1, z1

        rot_vals[i] = np.column_stack([x1[:2], y1[:2]])

    return rot_vals

clicked = False

def on_click(event):
    global clicked
    if event.inaxes != ax:
        return
    if event.button is MouseButton.LEFT:
        clicked = True

def on_release(event):
    global clicked
    if event.inaxes != ax:
        return
    clicked = False

def on_move(event):
    global joint_rotation
    redraw = False
    if event.inaxes != ax:
        return
    if clicked:
        if event.xdata is not None:
            xy = np.array([event.xdata, event.ydata])
            dist = 10
            idx = -1
            for i, p in enumerate(cur_point):
                d = np.linalg.norm(xy-p)
                if d < dist:
                    dist = d
                    idx = i
            if idx != -1:
                cur_point[0] = xy
                sc2.set_offsets(cur_point)
            redraw = True
    if redraw:
        global_fk_n_draw(interp(cur_point[0]))
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()