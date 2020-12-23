# -*- coding: utf-8 -*-

from import_data import zurich_data, ambica_data
from scipy.linalg import norm
from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from accel_calibration import align_gravity_vector, align_xy_axis
from scipy.signal import medfilt
from sklearn.linear_model import LinearRegression

def process_zurich_data():
    subj1 = zurich_data("s01")
    motion = "leg_rotation"
    accel_name = "ankle"

    rotated_accel_vector =  \
        align_gravity_vector(subj1.accel_data()[accel_name][motion], 5, 500)

    window = np.arange(1000, 2000, 1)
    x_dir = (subj1.accel_data()[accel_name][motion][0, window])
    y_dir = (subj1.accel_data()[accel_name][motion][1, window])
    rotation_angle = np.arctan2(y_dir, x_dir)

    plt.figure(num=2)

    plt.title("accel data, " + motion + ", " + accel_name)
    plt.plot(subj1.accel_data()[accel_name][motion][0, :-5], label="x-axis")
    plt.plot(subj1.accel_data()[accel_name][motion][1, :-5], label="y-axis")
    plt.plot(subj1.accel_data()[accel_name][motion][2, :-5], label="z-axis")
    plt.legend(loc="upper left")
    mplcursors.cursor()
    plt.show()

    plt.figure(num=3)

    plt.title("accel data, " + motion + ", " + accel_name)
    plt.plot(rotated_accel_vector[:-5], label=("x-axis", "y-axis", "z-axis"))
    # plt.plot(rotated_accel_vector[:-5], label="y-axis")
    # plt.plot(rotated_accel_vector[:-5], label="z-axis")
    plt.legend(loc="upper left")
    mplcursors.cursor()
    plt.show()


def process_ambica_data():
    leg_moving = ambica_data("leg_moving.TXT")
    accel_data = leg_moving.accel_data()
    accel_name = "ankle"
    accel_data[accel_name][:, 0] = medfilt(accel_data[accel_name][:, 0], 11)
    accel_data[accel_name][:, 1] = medfilt(accel_data[accel_name][:, 1], 11)
    accel_data[accel_name][:, 2] = medfilt(accel_data[accel_name][:, 2], 11)

    accel_data_z_axis_aligned =  \
        align_gravity_vector(accel_data[accel_name], 400, 1500)

    accel_data_aligned = align_xy_axis(accel_data_z_axis_aligned)

    plt.figure(1)
    plt.title("accel data, " + accel_name)
    plt.plot(accel_data[accel_name], label="axis")
    plt.legend(loc="upper left")
    mplcursors.cursor()
    plt.show()

    plt.figure(2)
    plt.title("rotated data, " + accel_name)
    plt.plot(accel_data_z_axis_aligned, label="axis")
    plt.legend(loc="upper left")
    mplcursors.cursor()
    plt.show()

    plt.figure(3)
    plt.title("aligned data, " + accel_name)
    plt.plot(accel_data_aligned)
    plt.legend(loc="upper left")
    mplcursors.cursor()
    plt.show()


if __name__ == "__main__":
    # process_zurich_data()
    process_ambica_data()

