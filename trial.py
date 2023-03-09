from skimage import transform
import numpy as np
import matplotlib.pyplot as plt

# Load an example image
img = plt.imread('src/docs/phantom images/T1.jpg')

# Define the shear angle in degrees
shear_angle = 30

# Convert the shear angle to radians
shear_angle_rad = np.deg2rad(shear_angle)

# Define the shear matrix
shear_matrix = np.array([[1, np.tan(shear_angle_rad), 0],
                         [0, 1, 0],
                         [0, 0, 1]])

# Apply the shear transformation to the image
sheared_img = transform.AffineTransform(matrix=shear_matrix)(img)

# Inverse transform the sheared image to obtain the k-space
k_space, inverse_matrix = transform.AffineTransform(matrix=shear_matrix).inverse(sheared_img)

# Plot the k-space magnitude
plt.imshow(np.abs(k_space), cmap='gray')
plt.show()
