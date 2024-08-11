import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression

def regularize_line(XY):
    # Fit a line to the points
    reg = LinearRegression().fit(XY[:, 0].reshape(-1, 1), XY[:, 1])
    line_X = np.linspace(np.min(XY[:, 0]), np.max(XY[:, 0]), len(XY))
    line_Y = reg.predict(line_X.reshape(-1, 1))
    return np.vstack((line_X, line_Y)).T

def regularize_circle(XY):
    # Find the center and radius of the circle
    center = np.mean(XY, axis=0)
    radius = np.mean(distance.cdist([center], XY)[0])
    angles = np.linspace(0, 2 * np.pi, len(XY))
    circle_X = center[0] + radius * np.cos(angles)
    circle_Y = center[1] + radius * np.sin(angles)
    return np.vstack((circle_X, circle_Y)).T

def regularize_rectangle(XY):
    # Fit a rectangle to the points
    min_x, min_y = np.min(XY, axis=0)
    max_x, max_y = np.max(XY, axis=0)
    rectangle_X = [min_x, max_x, max_x, min_x, min_x]
    rectangle_Y = [min_y, min_y, max_y, max_y, min_y]
    return np.vstack((rectangle_X, rectangle_Y)).T

def regularize_star(XY):
    # Placeholder for star shape regularization
    # Implement star shape regularization here
    return XY


def is_circle(points, tol=0.2):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.allclose(distances, distances[0], rtol=tol)

def is_regular_polygon(points, tol=0.2):
    vectors = np.diff(points, axis=0, append=points[:1])
    side_lengths = np.linalg.norm(vectors, axis=1)
    dot_products = np.einsum('ij,ij->i', vectors[:-1], vectors[1:])
    norms = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
    angles = np.arccos(np.clip(dot_products / norms, -1.0, 1.0))
    return np.allclose(side_lengths, side_lengths[0], rtol=tol) and np.allclose(angles, angles[0], rtol=tol)

def is_star_shape(points, tol=0.1):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    sorted_distances = np.sort(distances)
    # Assume a star shape has a significant range in distances
    if (sorted_distances[-1] - sorted_distances[0]) / sorted_distances[0] < 0.5:
        return False

    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    angles = np.sort(angles)
    angle_diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
    expected_diffs = np.pi / (len(points) // 2)

    # Additional check for alternating angles
    alternating_pattern = np.allclose(angle_diffs[::2], expected_diffs, atol=tol)
    return alternating_pattern

def regularize_star_shape_(XY):
    # Calculate the center of the points
    center = np.mean(XY, axis=0)
    # Calculate distances from the center and angles
    distances = np.linalg.norm(XY - center, axis=1)
    angles = np.arctan2(XY[:, 1] - center[1], XY[:, 0] - center[0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    sorted_distances = distances[sorted_indices]
    sorted_angles = angles[sorted_indices]
    
    # Identify peaks and valleys based on distance
    mean_distance = np.mean(sorted_distances)
    peaks = sorted_distances > mean_distance
    valleys = sorted_distances <= mean_distance
    
    peak_distances = sorted_distances[peaks]
    valley_distances = sorted_distances[valleys]
    
    mean_peak_distance = np.mean(peak_distances)
    mean_valley_distance = np.mean(valley_distances)
    
    regularized_points = []
    peak_idx = 0
    valley_idx = 0
    for i in range(len(sorted_angles)):
        angle = sorted_angles[i]
        if peaks[i % len(peaks)]:
            distance = mean_peak_distance
            peak_idx += 1
        else:
            distance = mean_valley_distance
            valley_idx += 1
        
        x = center[0] + distance * np.cos(angle)
        y = center[1] + distance * np.sin(angle)
        regularized_points.append([x, y])
    
    return np.array(regularized_points)


def identify_and_regularize(XY):
    if len(XY) < 3:
        return regularize_line(XY)
    
    distances = distance.pdist(XY, 'euclidean')
    if is_star_shape(XY):
        print("Star Shape Detected")
        return regularize_star_shape_(XY)
    
    elif is_circle(XY):
        print("Circle Detected")
        return regularize_circle(XY)
    
    else :
        return regularize_rectangle(XY)  # Add more shape identifications as needed

def regularise_curves(isolated_paths):
    regularized_paths = []
    for path in isolated_paths:
        regularized_path = []
        for points in path:
            regularized_points = identify_and_regularize(points)
            regularized_path.append(regularized_points)
        regularized_paths.append(regularized_path)
    return regularized_paths

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Read isolated paths
isolated_csv_path = '../problems/isolated.csv'
isolated_paths = read_csv(isolated_csv_path)

# Regularize the curves
regularized_paths = regularise_curves(isolated_paths)

# Plot the regularized curves
def plot(paths_XYs, title="", filename="plot.png"):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.savefig(filename)
    plt.close()
plot(isolated_paths, title="Isolated Curves", filename="isolated_plot.png")
plot(regularized_paths, title="Regularized Curves", filename="regularized_plot.png")