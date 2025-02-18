import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
from matplotlib.widgets import Slider
from functools import partial
import igl
import os

v_selected = None
f_d = None


def read_mesh(path):
    global v_selected
    global f_d
    
    v, f = igl.read_triangle_mesh(f'obj/{path}')
    if 'reference' in path:
        _, _, f_d, _, v_selected = igl.decimate(v, f, 5000)
        # print(v_selected.shape)
    
    return v[v_selected], f_d

blendshapes=[
    "eyeBlink_L.obj",
    "eyeBlink_R.obj",
    "mouthSmile_L.obj",
    "mouthSmile_R.obj",
]

bs_weights = np.array([
    0.6, 0.6, 0.5, 1.0
])

def calculate_blendshapes(bss, rv):
    bs = []
    rv = rv.copy().reshape(-1)

    for b in bss:
        b = b.copy().reshape(-1)
        bs.append(b - rv)
    
    return np.array(bs).T

def blendshape(rv, bs, weights):
    fv = rv.copy()
    fv = fv.reshape(-1)
    fv = fv + bs@weights
    
    return fv.reshape(-1, 3)
    
if __name__ == "__main__":
    rv, ict_f = read_mesh("reference.obj")
    
    bss = [rv]
    for i in blendshapes:
        bss.append(read_mesh(i)[0])
    
    bs = calculate_blendshapes(bss[1:], rv)
    fv = blendshape(rv, bs, bs_weights)
    
    axcolor='lightgoldenrodyellow'
    fig = plt.figure(tight_layout=True, figsize=(16, 9),)
    gs =  fig.add_gridspec(2, 1, height_ratios=(3, 1))
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.view_init(elev=4, azim=-4)
    ax.set_xlim(-5, 15)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_axis_off()
        
    ax.plot_trisurf(fv[:,2], fv[:,0], fv[:,1], triangles = ict_f, linewidth=0.2, antialiased=False)
    sliders = []
    def update(idx, value):
        bs_weights[idx] = value
        fv = blendshape(rv, bs, bs_weights)
        ax.clear()
        ax.plot_trisurf(fv[:,2], fv[:,0], fv[:,1], triangles = ict_f, linewidth=0.2, antialiased=False)
        ax.view_init(elev=4, azim=-4)
        ax.set_xlim(-5, 15)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_axis_off()


    for i, s in enumerate(blendshapes):
        axis = plt.axes([0.25, 0.2+i*-0.05, 0.6, 0.03], facecolor=axcolor)
        slider = Slider(
            ax=axis,
            label=s,
            valmin=0.0,
            valmax=1,
            valinit=bs_weights[i],
        )
        slider.on_changed(partial(update, i))
        sliders.append(slider)
    

    # ax2 = fig.add_subplot(gs[1])
    # ax2.clear()



    plt.show()

