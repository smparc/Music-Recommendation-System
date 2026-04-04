"""
svd.py
------
Singular Value Decomposition implemented from scratch.

For any matrix A (m x n):
    A = U * Sigma * V^T

where:
    U  (m x m) — left singular vectors (orthonormal)
    Σ  (m x n) — diagonal matrix of singular values (descending)
    V^T (n x n) — right singular vectors (orthonormal)

Strategy
--------
We compute the eigendecomposition of A^T A (symmetric, positive semi-definite),
which gives us V and the singular values. We then derive U from U_i = A v_i / sigma_i.
The eigendecomposition uses the QR algorithm with Gram-Schmidt orthogonalization.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper: Gram-Schmidt QR decomposition
# ---------------------------------------------------------------------------

def _gram_schmidt(A: np.ndarray):
    """
    QR decomposition via classical Gram-Schmidt orthogonalization.

    Parameters
    ----------
    A : np.ndarray of shape (m, n)

    Returns
    -------
    Q : np.ndarray (m, n) — orthonormal columns
    R : np.ndarray (n, n) — upper triangular
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = v  # zero vector, keep as-is
    return Q, R


def _qr_algorithm(S: np.ndarray, max_iter: int = 1000, tol: float = 1e-10):
    """
    Compute eigenvalues and eigenvectors of a symmetric matrix S
    via the QR iteration algorithm.

    Parameters
    ----------
    S : np.ndarray (n, n) — symmetric matrix
    max_iter : int
    tol : float — convergence tolerance

    Returns
    -------
    eigenvalues  : np.ndarray (n,)
    eigenvectors : np.ndarray (n, n)  — columns are eigenvectors
    """
    n = S.shape[0]
    A = S.copy().astype(float)
    V = np.eye(n)  # accumulate eigenvectors

    for _ in range(max_iter):
        Q, R = _gram_schmidt(A)
        A = R @ Q
        V = V @ Q

        # Check off-diagonal convergence
        off_diag = np.sum(np.abs(A - np.diag(np.diag(A))))
        if off_diag < tol:
            break

    eigenvalues = np.diag(A)
    eigenvectors = V
    return eigenvalues, eigenvectors


def svd(A: np.ndarray, k: int = None):
    """
    Compute the (truncated) SVD of matrix A.

    A = U Σ V^T

    Parameters
    ----------
    A : np.ndarray of shape (m, n)
    k : int or None
        Number of singular values/vectors to return.
        If None, returns full SVD.

    Returns
    -------
    U     : np.ndarray (m, k)
    sigma : np.ndarray (k,)   — singular values in descending order
    Vt    : np.ndarray (k, n) — rows are right singular vectors
    """
    m, n = A.shape

    # Step 1: compute A^T A  (n x n, symmetric)
    AtA = A.T @ A

    # Step 2: eigendecompose A^T A  ->  V, sigma^2
    eigenvalues, V = _qr_algorithm(AtA)

    # Clip small negatives caused by floating-point errors
    eigenvalues = np.clip(eigenvalues, 0, None)
    sigma_full = np.sqrt(eigenvalues)

    # Step 3: sort descending
    order = np.argsort(sigma_full)[::-1]
    sigma_full = sigma_full[order]
    V = V[:, order]

    # Step 4: keep only non-trivial singular values
    rank = int(np.sum(sigma_full > 1e-10))
    if k is None:
        k = rank
    else:
        k = min(k, rank)

    sigma = sigma_full[:k]
    Vk = V[:, :k]

    # Step 5: derive left singular vectors  U_i = A v_i / sigma_i
    U = np.zeros((m, k))
    for i in range(k):
        U[:, i] = (A @ Vk[:, i]) / sigma[i]

    Vt = Vk.T  # (k, n)

    return U, sigma, Vt


def explained_variance_ratio(sigma: np.ndarray) -> np.ndarray:
    """
    Compute the fraction of total variance explained by each singular value.

    Parameters
    ----------
    sigma : np.ndarray — singular values (descending)

    Returns
    -------
    np.ndarray — explained variance ratios
    """
    sigma_sq = sigma ** 2
    total = np.sum(sigma_sq)
    if total == 0:
        return np.zeros_like(sigma)
    return sigma_sq / total
