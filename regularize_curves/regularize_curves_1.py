import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Function to read CSV file and extract points
def read_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    paths = []
    unique_paths = np.unique(data[:, 0])
    for path_id in unique_paths:
        path_points = data[data[:, 0] == path_id][:, 1:]
        paths.append(path_points)
    return paths

# Function to detect straight lines
def is_straight_line(points, tolerance=1e-2):
    x = points[:, 0]
    y = points[:, 1]
    coeffs = np.polyfit(x, y, 1)
    line = np.polyval(coeffs, x)
    return np.all(np.abs(line - y) < tolerance)

# Function to detect circles (naive approach assuming perfect circle)
def is_circle(points, tolerance=1e-2):
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    return np.std(distances) < tolerance

# Function to detect rectangles (approximation)
def is_rectangle(points, tolerance=1e-2):
    hull = ConvexHull(points)
    if len(hull.vertices) == 4:
        edges = np.diff(points[hull.vertices], axis=0, append=points[hull.vertices][[0]])
        angles = np.arccos(np.dot(edges, np.roll(edges, 1, axis=0).T) / (np.linalg.norm(edges, axis=1) * np.linalg.norm(np.roll(edges, 1, axis=0), axis=1)))
        return np.allclose(angles, np.pi/2, atol=tolerance)
    return False

# Function to detect regular polygons (approximation)
def is_regular_polygon(points, tolerance=1e-2):
    hull = ConvexHull(points)
    distances = np.linalg.norm(np.diff(points[hull.vertices], axis=0, append=points[hull.vertices][[0]]), axis=1)
    return np.std(distances) < tolerance

# Function to detect star shapes (placeholder)
def is_star_shape(points):
    # Implement star shape detection logic
    return False

# Plot function
def plot(paths_XYs, title, save_path=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['blue', 'green', 'red', 'purple']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        ax.plot(XYs[:, 0], XYs[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
isolated_csv_path = '../problems/isolated.csv'
isolated_paths = read_csv(isolated_csv_path)

count = 0   
for points in isolated_paths:
    if is_straight_line(points):
        print("Straight Line Detected")
    elif is_circle(points):
        print("Circle Detected")
    elif is_rectangle(points):
        print("Rectangle Detected")
    elif is_regular_polygon(points):
        print("Regular Polygon Detected")
    elif is_star_shape(points):
        print("points ======> ", points)
        print("Star Shape Detected")
    else:
        print("Irregular Shape Detected")
    count += 1

print("Total number of paths: ", count)
plot(isolated_paths, "Isolated Curves", "./isolated_curves.png")
