import numpy as np

# 2D bases
Zero = np.array([[1, 0]], dtype=np.complex_).transpose()
One = np.array([[0, 1]], dtype=np.complex_).transpose()

factor = complex(1 / np.sqrt(2), 0.0)
Plus = factor * np.array([[1, 1]], dtype=np.complex_).transpose()
Minus = factor * np.array([[1, -1]], dtype=np.complex_).transpose()


# compute the conjugate transpose of the input matrix
def dagger(a: np.array) -> np.array:
    return a.transpose().conj()


# check if the input matrix is unitary
def is_unitary(a: np.array) -> bool:
    return np.allclose(np.linalg.inv(a), dagger(a))


# check if the input matrix is hermitian
def is_hermitian(a: np.array) -> bool:
    return np.allclose(a, dagger(a))
