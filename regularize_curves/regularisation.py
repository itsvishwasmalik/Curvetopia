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


def regularize_ellipse(XY, num_points=100):
    mean = np.mean(XY, axis=0)
    centered_points = XY - mean
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    minor_axis = eigenvectors[:, np.argmin(eigenvalues)]

    semi_major_axis_length = np.sqrt(np.max(eigenvalues))
    semi_minor_axis_length = np.sqrt(np.min(eigenvalues))

    theta = np.linspace(0, 2 * np.pi, num_points)
    ellipse_points = np.zeros((num_points, 2))
    for i in range(num_points):
        ellipse_points[i, 0] = semi_major_axis_length * np.cos(theta[i])
        ellipse_points[i, 1] = semi_minor_axis_length * np.sin(theta[i])

    ellipse_points = np.dot(ellipse_points, [major_axis, minor_axis]) + mean

    return ellipse_points


def is_line(points, tol=0.1):
    if len(points) < 3:
        return True
    slopes = np.diff(points, axis=0)
    slopes_norm = np.linalg.norm(slopes, axis=1, keepdims=True)
    if np.any(slopes_norm == 0):
        return False
    slopes = slopes / slopes_norm
    return np.allclose(slopes, slopes[0], atol=tol)


def is_circle(points, tol=0.2):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.allclose(distances, distances[0], rtol=tol)


def is_rectangle(points, tolerance=1e-2):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    primary_axis = eigenvectors[:, np.argmax(eigenvalues)]
    secondary_axis = eigenvectors[:, np.argmin(eigenvalues)]

    proj_primary = np.dot(centered_points, primary_axis)
    proj_secondary = np.dot(centered_points, secondary_axis)

    length = np.max(proj_primary) - np.min(proj_primary)
    width = np.max(proj_secondary) - np.min(proj_secondary)

    aspect_ratio = length / width if width != 0 else 0

    is_rectangular = (1.0 - tolerance) <= aspect_ratio <= (1.0 + tolerance)
    
    uniform_distribution = np.allclose(np.histogram(proj_primary, bins=10)[0],
                                       np.histogram(proj_secondary, bins=10)[0],
                                       atol=len(points) * 0.1)
    return True
    return is_rectangular and uniform_distribution

def is_ellipse(points, tol=0.1):
    mean = np.mean(points, axis=0)
    centered_points = points - mean
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    minor_axis = eigenvectors[:, np.argmin(eigenvalues)]
    axis_ratio = np.sqrt(np.max(eigenvalues) / np.min(eigenvalues))

    proj_major = np.dot(centered_points, major_axis)
    proj_minor = np.dot(centered_points, minor_axis)

    fit_quality_major = np.std(proj_major) / np.mean(np.abs(proj_major))
    fit_quality_minor = np.std(proj_minor) / np.mean(np.abs(proj_minor))

    return (1 - tol) < axis_ratio < (1 + tol) and \
           fit_quality_major < tol and \
           fit_quality_minor < tol

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


def draw_star_shape(center, inner_radius, outer_radius, rotation_angle):
    num_points = 5  # Number of star points
    angle_step = np.pi / num_points  # Angle between star points

    # Generate points for the star
    points = []
    for i in range(num_points * 2):
        angle = i * angle_step + np.radians(rotation_angle)
        if i % 2 == 0:
            r = outer_radius
        else:
            r = inner_radius
        x = center[0] + r * np.cos(angle)
        y = center[1] + r * np.sin(angle)
        points.append((x, y))

    # Add the first point at the end to close the star
    points.append(points[0])
    x_points, y_points = zip(*points)
    return np.array(points)


def regularize_star_shape_(XY):
    # detect the center of the star
    center = np.mean(XY, axis=0)
    # print("Center: ", center)

    # detect the outer radius of the star
    distances = np.linalg.norm(XY - center, axis=1)
    outer_radius = np.max(distances)
    # print("Outer Radius: ", outer_radius)

    # detect the inner radius of the star
    inner_radius = np.min(distances)
    # print("Inner Radius: ", inner_radius)

    # detect the rotation angle of the star from the x-axis
    angles = np.arctan2(XY[:, 1] - center[1], XY[:, 0] - center[0])
    angles = np.sort(angles)
    angle_diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
    expected_diffs = np.pi / (len(XY) // 2)
    rotation_angle = np.mean(angles)
    # print("Rotation Angle: ", rotation_angle)

    # draw the star shape
    star_points = draw_star_shape(center, inner_radius, outer_radius, np.degrees(rotation_angle))
    return star_points


def identify_and_regularize(XY):
    if len(XY) < 3:
        return regularize_line(XY)
    
    distances = distance.pdist(XY, 'euclidean')

    if is_line(XY):
        return regularize_line(XY)
    
    elif is_ellipse(XY):
        print("Ellipse Detected")
        return regularize_ellipse(XY)
    
    elif is_star_shape(XY):
        print("Star Shape Detected")
        return regularize_star_shape_(XY)
    
    elif is_circle(XY):
        print("Circle Detected")
        return regularize_circle(XY)
    
    elif is_rectangle(XY):
        print("Rectangle Detected")
        return regularize_rectangle(XY)  # Add more shape identifications as needed
    
    else:
        return XY

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
    plt.savefig("assets/" + filename)
    plt.close()

def save_shapes_for_symmetry(i, XYs, filename):
    # Saving without x and y axes
    # Calculate the range of the data
    x_min, x_max = min(XY[:, 0].min() for XY in XYs), max(XY[:, 0].max() for XY in XYs)
    y_min, y_max = min(XY[:, 1].min() for XY in XYs), max(XY[:, 1].max() for XY in XYs)

    # Calculate the figure size based on the range
    fig_width = (x_max - x_min) / 10  # Adjust the divisor to control the scaling
    fig_height = (y_max - y_min) / 10  # Adjust the divisor to control the scaling

    fig, ax = plt.subplots(tight_layout=True, figsize=(fig_width, fig_height))
    for XY in XYs:
        ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove x and y axes
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    plt.savefig("../symmetry/assets/images/" + filename + "_" + str(i) + ".svg", format='svg', transparent=True)
    plt.close()


def plot_shapes_seperately(paths_XYs, title="", filename="plot.png"):
    for i, XYs in enumerate(paths_XYs):
        fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
        ax.set_aspect('equal')
        plt.title(title + " " + str(i))
        plt.savefig("assets/" + filename + "_" + str(i) + ".png")
        plt.close()

        save_shapes_for_symmetry(i, XYs, filename)

plot(isolated_paths, title="Isolated Curves", filename="isolated_plot.png")
plot(regularized_paths, title="Regularized Curves", filename="regularized_plot.png")
plot_shapes_seperately(regularized_paths, title="Regularized Curves", filename="regularized_plot")

def convert_paths_to_csv(paths_XYs, filename="regularized.csv"):
    with open(filename, 'w') as f:
        for i, XYs in enumerate(paths_XYs):
            for j, XY in enumerate(XYs):
                for x, y in XY:
                    f.write(f"{i},{j},{x},{y}\n")
    print(f"Saved regularized paths to {filename}")

convert_paths_to_csv(regularized_paths, filename="regularized.csv")