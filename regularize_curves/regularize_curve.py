import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit

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

def plot(paths_XYs, title="", filename="plot.png"):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def is_straight_line(points, tol=0.1):
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

def is_ellipse(points, tol=0.2):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    distances = np.sort(distances)
    return np.allclose(distances, distances[0], rtol=tol)

def is_rectangle(points, tol=0.4):
    # if len(points) < 4:
    #     return False
    vectors = np.diff(points, axis=0)
    vectors = np.vstack([vectors, vectors[0]])
    dot_products = np.einsum('ij,ij->i', vectors[:-1], vectors[1:])
    norms = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
    angles = np.arccos(np.clip(dot_products / norms, -1.0, 1.0))
    return np.allclose(angles, np.pi / 2, atol=tol)

# Function to fit a circle to a set of points
def fit_circle(points):
    def circle_equation(x, a, b, r):
        return np.sqrt(r**2 - (x - a)**2) + b

    def objective(params, x, y):
        return circle_equation(x, *params) - y

    x = points[:, 0]
    y = points[:, 1]
    center_guess = np.mean(points, axis=0)
    radius_guess = np.max(np.linalg.norm(points - center_guess, axis=1))
    params, _ = curve_fit(circle_equation, x, y, p0=[center_guess[0], center_guess[1], radius_guess])
    return params

# Function to detect rounded rectangles
def is_rounded_rectangle(points, tolerance=1e-2):
    hull = ConvexHull(points)
    vertices = points[hull.vertices]
    if len(vertices) != 8:
        return False  # A rounded rectangle has four sides and four rounded corners

    # Separate corners and edges
    edges = []
    corners = []
    for i in range(4):
        edge = vertices[i*2:(i*2)+2]
        corner = vertices[(i*2+1)%8:(i*2+3)%8]
        edges.append(edge)
        corners.append(corner)

    # Check if edges are approximately straight lines
    for edge in edges:
        if not is_straight_line(edge, tolerance):
            return False

    # Check if corners are approximately circular
    for corner in corners:
        params = fit_circle(corner)
        a, b, r = params
        if not np.allclose(np.linalg.norm(corner - [a, b], axis=1), r, atol=tolerance):
            return False

    return True


def is_regular_polygon(points, tol=0.2):
    vectors = np.diff(points, axis=0, append=points[:1])
    side_lengths = np.linalg.norm(vectors, axis=1)
    dot_products = np.einsum('ij,ij->i', vectors[:-1], vectors[1:])
    norms = np.linalg.norm(vectors[:-1], axis=1) * np.linalg.norm(vectors[1:], axis=1)
    angles = np.arccos(np.clip(dot_products / norms, -1.0, 1.0))
    return np.allclose(side_lengths, side_lengths[0], rtol=tol) and np.allclose(angles, angles[0], rtol=tol)

def is_star_shape(points, tol=0.2):
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


def detect_rectangle(points):
    points = points.astype(np.int32)
    hull = cv2.convexHull(points)
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    scale = 5
    translate = 100
    scaled_points = points * scale + translate
    approx_scaled = approx * scale + translate

    for p in scaled_points:
        cv2.circle(img, tuple(p), 2, (255, 0, 0), -1)

    cv2.polylines(img, [approx_scaled], isClosed=True, color=(0, 255, 0), thickness=2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Hand-drawn Rectangle')
    plt.savefig('detected_rectangle.png')
    plt.close()

# Example usage:
isolated_csv_path = '../problems/isolated.csv'
isolated_paths = read_csv(isolated_csv_path)

count = 0   
for path in isolated_paths:
    for points in path:
        detect_rectangle(points)
        if is_straight_line(points):
            print("Straight Line Detected")
        elif is_circle(points):
            # print("points ======> ", list(points))
            print("Circle Detected")
        elif is_rectangle(points):
            print("Rectangle Detected")
        elif is_rounded_rectangle(points):
            print("Rounded Rectangle Detected")
        elif is_regular_polygon(points):
            print("Regular Polygon Detected")
        elif is_star_shape(points):
            # print("points ======> ", list(points))
            print("Star Shape Detected")
        else:
            # print("Points ======> ", list(points))
            print("Irregular Shape Detected")
        count += 1

print("Total number of paths: ", count)
# plot(isolated_paths, "Isolated Curves", "./isolated_curves.png")
# plot only third isolated path in the list
plot(isolated_paths, "Isolated Curves", "./isolated_curves.png")
