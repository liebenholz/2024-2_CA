import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

# set noise scale
ns = 0.2

def make_noise_linear(a, b, num=15): # ax + b = y
    xs = np.linspace(-4, 4, num) + np.random.randn(num) * ns
    f = lambda x: a*x +b
    ys = f(xs) + np.random.randn(num) * ns

    return xs, ys

def make_noise_quadratic(a, b, c, num=30): # ax^2 + bx + c = y
    xs = np.linspace(-4, 4, num) + np.random.randn(num) * ns
    f = lambda x: a*x**2.+ b*x + c
    ys = f(xs) + np.random.randn(num) * ns

    return xs, ys

def make_noise_sin(a, ω, φ, num = 30):# asin(ωx - φ)
    xs = np.linspace(-4, 4, num) + np.random.randn(num) * ns
    f = lambda x: a * np.sin(ω * x - φ)
    ys = f(xs) + np.random.randn(num) * (ns/2)

    return xs, ys

def make_noise_circle(x, y, r, num=30): # (x-a)^2 + (y-b)^2 = r^2
    xs = np.zeros(num, dtype=complex) + np.random.randn(num) * (ns/2)
    xs.imag = np.linspace(0, 2*np.pi, num)
    f = lambda _x: r*np.power(np.e, _x) + complex(x, y)

    xy = f(xs)
    xy.real += np.random.randn(num) * (ns/2)
    xy.imag += np.random.randn(num) * (ns/2)

    return xy.real, xy.imag

def solve_least_square_linear(x, y):
    '''
    f(x) = ax + b = y
    | x_1 1 | | a | = | y_1 |
    | x_2 1 | | b |   | y_2 |
    | x_3 1 |         | y_3 |
    '''
    A = np.empty([x.shape[0], 2])
    b = y
    A[:, 0] = x
    A[:, 1] = 1

    ab = np.linalg.solve(A.T@A, A.T@b) # A^T@A x = A^T b

    return lambda x: ab[0] * x + ab[1] # ax + b

def solve_least_square_quadratic(x, y):
    '''
    f(x) = ax^2 + bx + c = y
    | x_1^2 x_1 1 | | a | = | y_1 |
    | x_2^2 x_2 1 | | b |   | y_2 |
    | x_3^2 x_3 1 | | c |   | y_3 |
    '''
    A = np.empty([x.shape[0], 3])
    b = y
    # <--- code here --->
    A[:, 0] = x**2
    A[:, 1] = x
    A[:, 2] = 1
    # <--- end --->

    abc = np.linalg.solve(A.T@A, A.T@b) # A^T@A x = A^T b

    return lambda x: abc[0] * x**2. + abc[1] * x  + abc[2]

def solve_least_square_sin(x, y):
    a = 1
    ω = 1
    φ = 1

    for _ in range(50): # solve iterative
        '''
        f(x) = a*sin(ωx-φ) = y
        | ρf(x_1)/ρa f(x_1) ρf(x_1)/ρω ρf(x_1)/ρφ | | a | = | y_1 - f(x_1) |
        | ρf(x_2)/ρa f(x_2) ρf(x_2)/ρω ρf(x_2)/ρφ | | ω |   | y_2 - f(x_2) |
        | ρf(x_3)/ρa f(x_3) ρf(x_3)/ρω ρf(x_3)/ρφ | | φ |   | y_3 - f(x_3) |

        '''
        A = np.empty([x.shape[0], 3])

        # <--- code here --->
        A[:, 0] = np.sin(ω * x - φ)
        A[:, 1] = a * x * np.cos(ω * x - φ)
        A[:, 2] = -a * np.cos(ω * x - φ)
        # <--- end --->

        _y = lambda x : a * np.sin(ω * x - φ)
        b = y - _y(x)

        aωφ = np.linalg.solve(A.T@A, A.T@b) # A^T@A x = A^T b

        a += aωφ[0] * 0.2
        ω += aωφ[1] * 0.2
        φ += aωφ[2] * 0.2

        ...

    return lambda x : a * np.sin(ω*x - φ)

def solve_least_square_circle(x, y):
    '''
    (x-a)^2 + (y-b)^2 = r^2 => αx + βy + γ = x^2 + y^2 where α = 2a, β = 2b, γ = r^2 - a^2 - b^2
    | x_1 y_1 1 | | α | = | x_1^2 + y_1^2 |
    | x_2 y_2 1 | | β |   | x_2^2 + y_2^2 |
    | x_3 y_3 1 | | γ |   | x_3^2 + y_3^2 |
    '''

    A = np.empty([x.shape[0], 3])
    b = x**2 + y**2

    # <--- code here --->
    A[:, 0] = x
    A[:, 1] = y
    A[:, 2] = 1
    # <--- end --->

    abc = np.linalg.solve(A.T@A, A.T@b) # A^T@A x = A^T b

    α = abc[0]
    β = abc[1]
    γ = abc[2]

    x = α / 2
    y = β / 2
    r = np.sqrt(γ+x**2.+y**2.)

    return lambda _x: r*np.power(np.e, _x) + complex(x, y)


# plot
fig = plt.figure(figsize=(6,10), tight_layout=True)
gs = gridspec.GridSpec(4, 2,  height_ratios=[1, 1, 1, 0.3])
axs_lsq = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
ax_error = fig.add_subplot(gs[2, :])
titles = ['linear', 'quadratic', 'sin', 'circle']
for i, ax in enumerate(axs_lsq):
    ax.set_title(titles[i])
    ax.grid()
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
ax_error.grid()

# noisy points
x, y = make_noise_linear(1, 2)

# plot noisy points
scs = []
for ax in axs_lsq:
    sc = ax.scatter(x, y)
    scs.append(sc)

# L-east SQ-uare solving
linear = solve_least_square_linear(x, y)
xs = np.linspace(-4, 4, 100)
r_xs = np.zeros(100, dtype=complex)
r_xs.imag = np.linspace(0, 2*np.pi, 100)

lines = []
for ax in axs_lsq:
    line = ax.plot(xs, linear(xs), color='r', lw=3)[0]
    lines.append(line)

err_line = ax_error.plot([0, 1, 2, 3], [2,2,2,2], '-o')[0]
ax_error.set_title("residual error")
ax_error.set_xticks([0, 1, 2, 3], ['linear', 'quadratic', 'sin', 'circle'])

# draw silder
axcolor='lightgoldenrodyellow'
belowfig = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)
numsilder = Slider(
    ax=belowfig,
    label='num of points',
    valmin=10,
    valmax=10000,
    valinit=15,
)

from functools import partial

def update(val):
    num = int(numsilder.val)

    # polynomials
    residuals = []
    for line, sc, noise_func, solve_func in zip(lines, scs,
        [partial(make_noise_linear, 1, 2), partial(make_noise_quadratic, 0.5, 0, -4), partial(make_noise_sin, 3, 2, 1,)],
        [solve_least_square_linear, solve_least_square_quadratic, solve_least_square_sin]):
        # make points
        x, y = noise_func(num)
        sc.set_offsets(np.stack([x, y], axis=-1))
        func = solve_func(x, y)
        residuals.append(np.mean((func(x)-y)**2.))
        line.set_ydata(func(xs))

    # circle
    x, y = make_noise_circle(0.5, 1, 2, num)
    scs[-1].set_offsets(np.stack([x, y], axis=-1))
    cif = solve_least_square_circle(x, y)
    cy = cif(r_xs)
    lines[-1].set_xdata(cy.real)
    lines[-1].set_ydata(cy.imag)
    t_xs = np.zeros(num, dtype=complex)
    t_xs.imag = np.linspace(0, 2*np.pi, num)
    cy = cif(t_xs)
    residuals.append(np.mean((x - cy.real)**2. + (y - cy.imag)**2.))

    err_line.set_ydata(residuals)
    ax_error.relim()
    ax_error.autoscale_view()
    fig.canvas.draw_idle()

update(0)
numsilder.on_changed(update)

plt.show()