import numpy as np

# Covariance matrix for yield log-returns
cov_yield = np.array([
    [0.001879, 0.000325, 0.000419, 0.000568, 0.000436, 0.000436, 0.000543, 0.000578, 0.000483, 0.000501],
    [0.000325, 0.000374, 0.000374, 0.000435, 0.000458, 0.000430, 0.000400, 0.000423, 0.000425, 0.000411],
    [0.000419, 0.000374, 0.000552, 0.000497, 0.000522, 0.000485, 0.000471, 0.000475, 0.000505, 0.000467],
    [0.000568, 0.000435, 0.000497, 0.000566, 0.000583, 0.000546, 0.000498, 0.000515, 0.000527, 0.000509],
    [0.000436, 0.000458, 0.000522, 0.000583, 0.000624, 0.000578, 0.000507, 0.000518, 0.000545, 0.000524],
    [0.000436, 0.000430, 0.000485, 0.000546, 0.000578, 0.000541, 0.000478, 0.000499, 0.000521, 0.000499],
    [0.000543, 0.000400, 0.000471, 0.000498, 0.000507, 0.000478, 0.000502, 0.000538, 0.000518, 0.000483],
    [0.000578, 0.000423, 0.000475, 0.000515, 0.000518, 0.000499, 0.000538, 0.000608, 0.000573, 0.000526],
    [0.000483, 0.000425, 0.000505, 0.000527, 0.000545, 0.000521, 0.000518, 0.000573, 0.000571, 0.000525],
    [0.000501, 0.000411, 0.000467, 0.000509, 0.000524, 0.000499, 0.000483, 0.000526, 0.000525, 0.000495]
])

# Covariance matrix for forward rate log-returns
cov_forward_rate = np.array([
    [0.000692, 0.000261, 0.000685, 0.000822],
    [0.000261, 0.002131, -0.000310, 0.000800],
    [0.000685, -0.000310, 0.002301, 0.000809],
    [0.000822, 0.000800, 0.000809, 0.001343]
])

# Calculate eigenvalues and eigenvectors for yield covariance matrix
eigenvalues_yield, eigenvectors_yield = np.linalg.eigh(cov_yield)

# Calculate eigenvalues and eigenvectors for forward rate covariance matrix
eigenvalues_forward_rate, eigenvectors_forward_rate = np.linalg.eigh(cov_forward_rate)

# Get the first eigenvalue and eigenvector (in terms of size) for both matrices
largest_eigenvalue_yield = eigenvalues_yield[-1]
largest_eigenvector_yield = eigenvectors_yield[:, -1]

largest_eigenvalue_forward_rate = eigenvalues_forward_rate[-1]
largest_eigenvector_forward_rate = eigenvectors_forward_rate[:, -1]

# Print results
print(f"Largest eigenvalue for yield covariance matrix: {largest_eigenvalue_yield}")
print(f"Largest eigenvector for yield covariance matrix: {largest_eigenvector_yield}")
print(f"Largest eigenvalue for forward rate covariance matrix: {largest_eigenvalue_forward_rate}")
print(f"Largest eigenvector for forward rate covariance matrix: {largest_eigenvector_forward_rate}")
