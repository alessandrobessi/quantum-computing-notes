import math
import numpy as np

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


# compute the conjugate transpose of the input matrix
def dagger(a: np.array) -> np.array:
    return a.transpose().conj()


# check if the input matrix is unitary
def is_unitary(a: np.array) -> bool:
    return np.allclose(np.linalg.inv(a), dagger(a))


# check if the input matrix is hermitian
def is_hermitian(a: np.array) -> bool:
    return np.allclose(a, dagger(a))


def info(a: np.array, name: str = None) -> None:
    if name is not None:
        print(name)
    print("shape:", a.shape)
    print(np.round(a, 3))
