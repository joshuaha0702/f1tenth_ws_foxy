import numpy as np
import math
from numba import njit
import yaml
from types import SimpleNamespace as Namespace
import os


@njit(cache=True)
def avgPoint(vertices):
    return np.sum(vertices, axis=0) / vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def closestPoint2Origin(a, b):
    ab = b - a
    ao = -a
    length = ab.dot(ab)
    if length < 1e-10:
        return a
    frac = ao.dot(ab) / length
    if frac < 0:
        return a
    if frac > 1:
        return b
    return frac * ab + a


@njit(cache=True)
def distance(vertices1, vertices2, direc):
    vertices1 = np.ascontiguousarray(vertices1)
    vertices2 = np.ascontiguousarray(vertices2)
    direc = np.ascontiguousarray(direc)
    a = support(vertices1, vertices2, direc)
    b = support(vertices1, vertices2, -direc)
    d = closestPoint2Origin(a, b)
    dist = np.linalg.norm(d)
    while True:
        if dist < 1e-10:
            return dist
        d = -d
        c = support(vertices1, vertices2, d)
        temp1 = c.dot(d)
        temp2 = a.dot(d)
        if (temp1 - temp2) < 1e-10:
            return dist
        p1 = closestPoint2Origin(a, c)
        p2 = closestPoint2Origin(c, b)
        dist1 = np.linalg.norm(p1)
        dist2 = np.linalg.norm(p2)
        if dist1 < dist2:
            b = c
            d = p1
            dist = dist1
        else:
            a = c
            d = p2
            dist = dist2


@njit(cache=True)
def get_vertices(pose, length, width):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    x, y = pose[0], pose[1]
    tl_x = -length / 2 * c + width / 2 * (-s) + x
    tl_y = -length / 2 * s + width / 2 * c + y
    tr_x = length / 2 * c + width / 2 * (-s) + x
    tr_y = length / 2 * s + width / 2 * c + y
    bl_x = -length / 2 * c + (-width / 2) * (-s) + x
    bl_y = -length / 2 * s + (-width / 2) * c + y
    br_x = length / 2 * c + (-width / 2) * (-s) + x
    br_y = length / 2 * s + (-width / 2) * c + y
    vertices = np.asarray([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
    return vertices


@njit(cache=True)
def nearest_point(point, trajectory):
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)
        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            continue
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start
            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c
            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
    return first_p, first_i, first_t


@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))


@njit(cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi
    return angle


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * math.pi:
        return angle - 2.0 * math.pi
    if angle < 0:
        return angle + 2.0 * math.pi
    return angle


def sample_traj(clothoid, npts, v):
    traj = np.empty((npts, 5))
    for i in range(npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = v
        traj[i, 3] = clothoid.Theta(s)
        traj[i, 4] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)
    return traj


@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    x_trans = x - orig_x
    y_trans = y - orig_y
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)
    return r, c


@njit(cache=True)
def map_collision(points, dt, map_metainfo, eps=0.4):
    orig_x, orig_y, orig_c, orig_s, height, width, resolution = map_metainfo
    collisions = np.empty((points.shape[0],))
    for i in range(points.shape[0]):
        r, c = xy_2_rc(points[i, 0], points[i, 1], orig_x, orig_y, orig_c, orig_s, height, width, resolution)
        if r < 0 or c < 0:
            collisions[i] = True
        else:
            if dt[r, c] <= eps:
                collisions[i] = True
            else:
                collisions[i] = False
    return np.ascontiguousarray(collisions)


@njit(cache=True)
def tripleProduct(a, b, c):
    ac = a.dot(c)
    bc = b.dot(c)
    return b * ac - a * bc


@njit(cache=True)
def perpendicular(pt):
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1 * temp
    return pt


@njit(cache=True)
def collision(vertices1, vertices2):
    index = 0
    simplex = np.empty((3, 2))
    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)
    d = position1 - position2
    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0
    a = support(vertices1, vertices2, d)
    simplex[index, :] = a
    if d.dot(a) <= 0:
        return False
    d = -a
    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False
        ao = -a
        if index < 2:
            b = simplex[0, :]
            ab = b - a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue
        b = simplex[1, :]
        c = simplex[0, :]
        ab = b - a
        ac = c - a
        acperp = tripleProduct(ab, ac, ac)
        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp
        simplex[1, :] = simplex[2, :]
        index -= 1
        iter_count += 1
    return False


@njit(cache=True)
def get_actuation_PD(pose_theta, lookahead_point, position, lookahead_distance, wheelbase, prev_error, P, D):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    error = 2.0 * waypoint_y / lookahead_distance ** 2
    if np.abs(waypoint_y) < 1e-4:
        return speed, 0., error
    steering_angle = P * error + D * (error - prev_error)
    return speed, steering_angle, error


def load_config(config_path):
    with open(config_path) as f:
        config_dict = yaml.full_load(f)
    return Namespace(**config_dict)
