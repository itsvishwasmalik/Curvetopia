import numpy as np
import matplotlib.pyplot as plt

# Example coordinates of the irregular star
coordinates = np.array([
    [2, 3], [4, 7], [5, 1], [8, 3], [6, 4],
    [7, 8], [3, 6], [1, 4], [2, 5], [4, 2]
])

# Step 1: Calculate the centroid
centroid = np.mean(coordinates, axis=0)

# Step 2: Calculate distances from centroid
distances = np.linalg.norm(coordinates - centroid, axis=1)

# Step 3: Identify the outer vertices (let's say we want 5 outer vertices)
num_outer_vertices = 5
outer_indices = np.argsort(distances)[-num_outer_vertices:]
outer_vertices = coordinates[outer_indices]

# Step 4: Regularize the star (for demonstration, we use the sorted outer vertices)
# In a real case, you might need more complex calculations for a precise regular star
sorted_vertices = sorted(outer_vertices, key=lambda v: np.arctan2(v[1] - centroid[1], v[0] - centroid[0]))

# Step 5: Draw the star
sorted_vertices = np.array(sorted_vertices)
star_indices = np.arange(len(sorted_vertices))
star_indices = np.concatenate([star_indices, [star_indices[0]]])  # Close the loop

plt.figure()
plt.plot(sorted_vertices[star_indices, 0], sorted_vertices[star_indices, 1], 'o-')
plt.fill(sorted_vertices[star_indices, 0], sorted_vertices[star_indices, 1], alpha=0.3)
plt.scatter(*centroid, color='red')  # Centroid
plt.title('Regularized Star Shape')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()