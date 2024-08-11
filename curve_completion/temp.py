import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# Load CSV data using your existing function
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

# Plot shapes using your existing function
def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Find closest points between two curves
def find_closest_points(curve1, curve2):
    dist_matrix = cdist(curve1, curve2)
    min_dist_indices = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    return curve1[min_dist_indices[0]], curve2[min_dist_indices[1]]

# Fit spline to curve
def fit_spline(curve, num_points=100):
    tck, u = splprep([curve[:, 0], curve[:, 1]], s=0)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack((x_new, y_new), axis=-1)

# Connect two curves using a smooth Bézier curve
def connect_curves(curve1, curve2, num_points=50):
    p1, p2 = find_closest_points(curve1, curve2)
    
    # Bézier curve control points
    def bezier_curve(t, p0, p1, p2, p3):
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # Objective function to minimize curvature at the endpoints
    def objective(ctrl_pts):
        p_ctrl = np.array(ctrl_pts).reshape(2, 2)
        curve = bezier_curve(np.linspace(0, 1, num_points), p1, p_ctrl[0], p_ctrl[1], p2)
        curvature = np.abs(np.gradient(np.gradient(curve[:, 1]))) + np.abs(np.gradient(np.gradient(curve[:, 0])))
        return np.sum(curvature)

    # Optimize control points
    initial_ctrl_pts = np.array([p1, p2]) + 0.1
    result = minimize(objective, initial_ctrl_pts.flatten(), method='L-BFGS-B')
    opt_ctrl_pts = result.x.reshape(2, 2)

    bezier_curve_points = bezier_curve(np.linspace(0, 1, num_points), p1, opt_ctrl_pts[0], opt_ctrl_pts[1], p2)
    
    return bezier_curve_points

# Complete the shape
def complete_shape(paths_XYs):
    completed_shapes = []
    for XYs in paths_XYs:
        for i in range(len(XYs) - 1):
            # Fit spline to each curve segment
            curve1 = fit_spline(XYs[i])
            curve2 = fit_spline(XYs[i + 1])
            
            # Connect the curves
            connection = connect_curves(curve1, curve2)
            
            # Combine the curves and connection
            completed_shape = np.vstack((curve1, connection, curve2))
            completed_shapes.append([completed_shape])
    
    return completed_shapes

# Main execution
occlusion2_csv_path = '/mnt/data/problems_unzipped/problems/occlusion2.csv'  # Update the path accordingly
paths_XYs = read_csv(occlusion2_csv_path)
plot(paths_XYs)  # Plot original shapes

completed_shapes = complete_shape(paths_XYs)
plot(completed_shapes)  # Plot completed shapes
