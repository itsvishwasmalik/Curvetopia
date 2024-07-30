import numpy as np

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

def has_reflection_symmetry(points):
    center = np.mean(points, axis=0)
    reflected_points = 2 * center - points
    return np.allclose(np.sort(points, axis=0), np.sort(reflected_points, axis=0))

def has_radial_symmetry(points):
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    angles = np.sort(angles)
    angle_diffs = np.diff(np.concatenate([angles, [angles[0] + 2 * np.pi]]))
    return np.allclose(angle_diffs, 2 * np.pi / len(points))

isolated_csv_path = '../problems/isolated.csv'
isolated_paths = read_csv(isolated_csv_path)

# Example usage:
for path in isolated_paths:
    print("path ======> ", path)
    for points in path:
        if has_reflection_symmetry(points):
            print("Reflection Symmetry Detected")
        if has_radial_symmetry(points):
            print("Radial Symmetry Detected")
