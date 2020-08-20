import math
import cmath
import numpy as np
from typing import Tuple

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# 2D bases
Zero = np.array([[1, 0]], dtype=np.complex_).transpose()
One = np.array([[0, 1]], dtype=np.complex_).transpose()

factor = complex(1 / np.sqrt(2), 0.0)
Plus = factor * np.array([[1, 1]], dtype=np.complex_).transpose()
Minus = factor * np.array([[1, -1]], dtype=np.complex_).transpose()

# 2D Gates
eye = np.array([[1, 0], [0, 1]], dtype=np.complex_)

# Pauli gates
X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex_)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)

# Hadamard gate
H = factor * np.array([[1, 1], [1, -1]], dtype=np.complex_)


# Phase Shift gate
def R(phi: np.float64) -> np.array:
    c = np.exp(complex(0, 1) * phi)
    return np.array([[1, 0], [0, c]], dtype=np.complex_)


# 4D Gates

# CNOT gate
cx = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
CX = np.array(cx, dtype=np.complex_)


# compute the conjugate transpose of the input matrix
def dagger(a: np.array) -> np.array:
    return a.transpose().conj()


# check if the input matrix is unitary
def is_unitary(a: np.array) -> bool:
    return np.allclose(np.linalg.inv(a), dagger(a))


# check if the input matrix is hermitian
def is_hermitian(a: np.array) -> bool:
    return np.allclose(a, dagger(a))


# pretty print arrays
def info(a: np.array, name: str = None) -> None:
    if name is not None:
        print(name)
    print("shape:", a.shape)
    print(np.round(a, 3))


# normalize an array
def normalize(a: np.array) -> np.array:
    norm = np.linalg.norm(a)
    if norm == 0:
        return a
    return a / norm


# get coefficients of a given qubit
def get_coefficients(a: np.array) -> Tuple[complex, complex]:
    return a[0][0], a[1][0]


# get polar coordinates of a qubit
# |q⟩ = a|0⟩ + b|1⟩ = cos(theta/2)|0⟩ + exp(i * phi) * sin(theta/2)|1⟩
def get_polar_coordinates(qubit: np.array) -> Tuple[float, float]:
    a, b = get_coefficients(qubit)
    theta = 2 * cmath.acos(a)
    if abs(cmath.sin(theta / 2)) > 1e-33:
        phi = 1 / complex(0, 1) * cmath.log(b / cmath.sin(theta / 2))
        return theta.real, phi.real

    return theta.real, complex(np.pi, 0).real


# get polar coordinates of a qubit
# r=1 because we are on a unit sphere
def get_cartesian_coordinates(qubit: np.array) -> Tuple[float, float, float]:
    theta, phi = get_polar_coordinates(qubit)
    r = 1
    x = r * cmath.sin(theta) * cmath.cos(phi)
    y = r * cmath.sin(theta) * cmath.sin(phi)
    z = r * cmath.cos(theta)
    return x.real, y.real, z.real


def draw_bloch_sphere(qubit: np.array) -> None:
    x, y, z = get_cartesian_coordinates(qubit)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    # ax.set_aspect("equal")

    # draw sphere
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
    sx = np.cos(u) * np.sin(v)
    sy = np.sin(u) * np.sin(v)
    sz = np.cos(v)
    ax.plot_wireframe(sx, sy, sz, color="k", alpha=0.1)

    a = Arrow3D(
        [0, x], [0, y], [0, z], mutation_scale=20, lw=2, arrowstyle="-|>", color="r"
    )
    ax.add_artist(a)
    plt.show()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
