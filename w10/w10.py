import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ss

from scipy.spatial import distance
from scipy.sparse.linalg import factorized
from matplotlib.backend_bases import MouseButton

num_points = 100

x = np.linspace(-5, 5, num_points)
y = np.reciprocal(np.abs(np.linspace(-3, 3, num_points))+1) * np.sin(np.linspace(0, 16*np.pi, num_points))

constraints = [0, num_points//2, num_points-1]
constpoints = np.stack([x[constraints], y[constraints]], axis=-1)

def make_A():
    A = ss.lil_matrix((num_points+len(constraints), num_points))

    for i in range(num_points):
        A[i, i] = 1
        if i > 0:
            A[i, i-1] = -0.5
        if i < num_points-1:
            A[i, i+1] = -0.5
        if i == 0 :
            A[i, i+1] = -1
        if i == num_points-1:
            A[i, i-1] = -1

    for i, c in enumerate(constraints):
        A[num_points+i, c] = 1

    return A.tocsc()

def make_b(constraints):
    b_x = np.zeros(num_points)
    b_y = np.zeros(num_points)

    for i in range(num_points):
        b_x[i] += x[i]
        b_y[i] += y[i]
        if i == 0:
            b_x[i] -= x[i+1]
            b_y[i] -= y[i+1]
        elif i == num_points-1:
            b_x[i] -= x[i-1]
            b_y[i] -= y[i-1]
        else:
                if i > 0:
                    b_x[i] += -0.5 * x[i-1]
                    b_y[i] += -0.5 * y[i-1]
                if i < num_points-1:
                    b_x[i] += -0.5 * x[i+1]
                    b_y[i] += -0.5 * y[i+1]

    const_xb = A.T @ np.concatenate([b_x, constraints[:, 0]])
    const_yb = A.T @ np.concatenate([b_y, constraints[:, 1]])

    # print(const_xb, const_yb)
    return const_xb, const_yb

A = make_A()
solve = factorized(A.T@A)

fig, ax = plt.subplots(figsize=(6, 6))

plt.grid()
plt.xlim(-5.5, 5.5)
plt.ylim(-5.5, 5.5)

ax.plot(x, y, "-", c='k', lw=5)[0]
l = ax.plot(x, y, "-", c='b', lw=5)[0]
point = ax.scatter(x[constraints], y[constraints], c='r', zorder=13)

is_pressed = False
picked_idx = -1

def find_picking_point(xy):
    global picked_idx
    threshhold = 3

    dist = distance.cdist(constpoints, [xy])
    idx = np.argmin(dist)
    min_v = dist[idx]

    picked_idx = idx if min_v <= threshhold else -1

def on_click(event):
    global is_pressed
    if event.button == MouseButton.LEFT:
        if event.xdata != None:
            is_pressed = True
            xy = np.array([event.xdata, event.ydata])
            find_picking_point(xy)

def on_move(event):
    if is_pressed and event.xdata != None:
        if picked_idx != -1:
            constpoints[picked_idx] = np.array([event.xdata, event.ydata])
            point.set_offsets(constpoints)
            c_xb, c_yb = make_b(constpoints)

            nx = solve(c_xb)
            ny = solve(c_yb)

            l.set_xdata(nx)
            l.set_ydata(ny)
            fig.canvas.draw()

def on_release(event):
    global is_pressed
    if event.button == MouseButton.LEFT:
        if event.xdata != None:
            is_pressed = False

plt.connect('button_press_event', on_click)
plt.connect('motion_notify_event', on_move)
plt.connect('button_release_event', on_release)

plt.show()