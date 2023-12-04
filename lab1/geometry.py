import numpy as np
import numpy.linalg as LA
from einops import rearrange
from typing import List, Union, Tuple
import numpy as np
from pyquaternion import Quaternion


Vector = Union[List[float], List[int], np.ndarray]


def to_numpy(l: Vector) -> np.ndarray:
    if isinstance(l, list):
        return np.array(l)
    elif isinstance(l, np.ndarray):
        return l
    else:
        raise NotImplementedError(f"{type(l)} is not supported")


def to_quaternion(q: Union[Vector, Quaternion]) -> Quaternion:
    if isinstance(q, list):
        assert len(q) == 4, f"{len(q)} != 4"
        return Quaternion(q)
    elif isinstance(q, np.ndarray):
        if q.shape == (4,):
            return Quaternion(q)
        elif q.shape == (3, 3) or q.shape == (4, 4):
            return Quaternion(matrix=q)
        else:
            raise ValueError(f"{q.shape} is neither a quaternion nor a rotation matrix")
    elif isinstance(q, Quaternion):
        return q
    else:
        raise NotImplementedError(f"{type(q)} is not supported")


def make_tf(translation: Vector, rotation: Union[Vector, Quaternion, np.ndarray]) -> np.ndarray:
    """
    Create a homogeneous transformation matrix
    :param translation: (3) - t_x, t_y, t_z
    :param rotation: either 4 number representing a quaternion, a Quaternion, or a rotation matrix
    :return: (4, 4)
    """
    tf = np.eye(4)
    tf[:3, -1] = to_numpy(translation)
    if isinstance(rotation, np.ndarray):
        if rotation.shape == (3, 3) or rotation.shape == (4, 4):
            tf[:3, :3] = rotation[:3, :3]
        else:
            raise ValueError(f"rotation has an invalid shape {rotation.shape}")
    else:
        tf[:3, :3] = to_quaternion(rotation).rotation_matrix
    return tf


def apply_tf(tf: np.ndarray, points: np.ndarray, in_place=False) -> Union[np.ndarray, None]:
    """
    Apply a homogeneous transformation to a set pof points
    :param tf: (4, 4) - transformation matrix
    :param points: (N, 3[+C]) - x, y, z, [C-dim features]
    :param in_place: to overwrite points' coordinate with the output or not.
        If True, this function doesn't return anything. Default: False
    :return:  (N, 3) - transformed coordinate
    """
    assert tf.shape == (4, 4), f"{tf.shape} is not a homogeneous transfomration matrix"
    assert points.shape[1] >= 3, f'expect points has at least 3 coord, get: {points.shape[1]}'
    xyz1 = np.pad(points[:, :3], pad_width=[(0, 0), (0, 1)], constant_values=1)  # (N, 4)
    xyz1 = rearrange(tf @ rearrange(xyz1, 'N C -> C N', C=4), 'C N -> N C')
    if in_place:
        points[:, :3] = xyz1[:, :3]
        return
    else:
        return xyz1[:, :3]


def get_points_in_range(points: np.ndarray, point_cloud_range: np.ndarray) -> np.ndarray:
    """
    Get points inside a limit.
    :param points: (N, 3[+C]) - x, y, z, [C-dim features]
    :param point_cloud_range: (6) - x_min, y_min, z_min, x_max, y_max, z_max
    :return: (N', 3[+C]) - x, y, z, [C-dim features]
    """
    assert points.shape[1] >= 3, f'expect points has at least 3 coord, get: {points.shape[1]}'
    assert point_cloud_range.shape[0] == 6, f"{point_cloud_range.shape[1]} != 6"
    # TODO
    points_inside = np.zeros([3, points.shape[1]])  # This is just a dummy value
    return points_inside


def check_points_in_range(points: np.ndarray, point_cloud_range: np.ndarray) -> bool:
    """
    Check if every points is in a limit.
    :param points: (N, 3[+C]) - x, y, z, [C-dim features]
    :param point_cloud_range: (6) - x_min, y_min, z_min, x_max, y_max, z_max
    :return: True if every point is within the limit, False otherwise
    """
    assert isinstance(point_cloud_range, type(points)), f"{type(point_cloud_range)} != {type(points)}"
    # TODO
    is_all_inside = False  # This is just a dummy value
    return is_all_inside


def orthogonal_projection(points: np.ndarray, point_cloud_range: np.ndarray, resolution: float) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute points' coordinate in bird-eye view using orthogonal projection.
    :param points: (N, 3[+C]) - x, y, z, [C-dim features]
    :param point_cloud_range: (6) - x_min, y_min, z_min, x_max, y_max, z_max
    :param resolution: size of a pixel measured by meter. Not distinguish between height & width because we assume
        pixels are square
    :return:
        - pixels_coord: (N_pixels, 2) - pixel_x (horizontal), pixel_y (vertical) | coordinate of occupied pixels
        - idx_pixels_to_points: (N,) - idx_pixels_to_points[i] = j means points[i] lands inside pixels_coord[j]
    """
    assert points.shape[1] >= 3, f'expect points has at least 3 coord, get: {points.shape[1]}'
    assert check_points_in_range(points, point_cloud_range), "some points are outside of range"

    # TODO
    pixels_coord = np.zeros((3, 2))  # This is just a dummy value
    idx_pixels_to_points = np.ones(points.shape[0])  # This is just a dummy value
    return pixels_coord, idx_pixels_to_points


def perspective_projection(points: np.ndarray, camera_intrinsic: np.ndarray) -> np.ndarray:
    """
    Apply pin-hole camera model to compute points' coordinate on images
    :param points: (N, 3) - x, y, z, in CAMERA's frame
    :param camera_intrinsic: (3, 3)
    :return:
        - pixels_coord: (N, 2) - pixel_x (horizontal), pixel_y (vertical)
    """
    assert points.shape[1] == 3, f"expect (N, 3), got {points.shape}"
    assert camera_intrinsic.shape == (3, 3), f"expect (3, 3), got {camera_intrinsic.shape}"
    points_in_pixel = rearrange(camera_intrinsic @ rearrange(points, 'N C -> C N', C=3),
                                'C N -> N C')
    # normalize
    points_in_pixel = points_in_pixel / points_in_pixel[:, [2]]
    return points_in_pixel[:, :2]


def rot_z(yaw: float) -> np.ndarray:
    """
    Create rotation matrix around z
    :param yaw:
    :return: (3, 3)
    """
    cos, sin = np.cos(yaw), np.sin(yaw)
    return np.array([
        cos, -sin,  0,
        sin,  cos,  0,
        0,      0,  1
    ]).reshape(3, 3).astype(float)


def find_points_in_boxes(points: np.ndarray, boxes: np.ndarray, tol=1e-2) -> np.ndarray:
    """
    Find points inside boxes. Note: points and boxes must be in the same frame.
    :param points: (N, 3[+C]) - x, y, z, [C-dim features]
    :param boxes: (B, 7[+D]) - center_x, center_y, center_z, dx, dy, dz, yaw, [velocity_x, velocity_y,...]
    :return:
        - boxes_to_points: (N,) - boxes_to_points[i] = j >=0 means points[i] is in boxes[j],
            boxes_to_points[i] == -1 means points[i] does not belong to any boxes
    """
    n_points, n_boxes = points.shape[0], boxes.shape[0]
    assert n_points > 0

    boxes_to_points = -np.ones(n_points)  # (N,)
    if n_boxes == 0:
        return boxes_to_points.astype(int)

    # TODO
    return boxes_to_points.astype(int)


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

