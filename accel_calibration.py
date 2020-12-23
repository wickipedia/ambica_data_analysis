# -*- coding: utf-8 -*-
from scipy.linalg import norm
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.linear_model import LinearRegression


def align_gravity_vector(data, window_start, window_end):
    mean_window = np.arange(window_start, window_end, 1)
    mean_x_dir = np.mean(data[mean_window, 0])
    mean_y_dir = np.mean(data[mean_window, 1])
    mean_z_dir = np.mean(data[mean_window, 2])

    z_axis = -np.array([0, 0, 1])
    gravity_vector = np.array([mean_x_dir, mean_y_dir, mean_z_dir])

    rotation_vector = np.cross(gravity_vector/norm(gravity_vector), z_axis)
    rotation_vector /= norm(rotation_vector)

    rotation_angle = np.arccos(mean_z_dir/norm(gravity_vector))

    r = R.from_rotvec(-rotation_angle * rotation_vector)
    return r.apply(data)


def align_xy_axis(data, window_start=0, window_end=-1):
    x_dir = data[window_start:window_end, 0].reshape((-1, 1))
    y_dir = data[window_start:window_end, 1]

    model = LinearRegression(fit_intercept=False).fit(x_dir, y_dir)
    rotation_angle = np.arctan(-model.coef_)
    rotation_vector = np.array([0, 0, 1])
    
    r = R.from_rotvec(rotation_angle * rotation_vector)

    return r.apply(data)
