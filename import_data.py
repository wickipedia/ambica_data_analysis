# -*- coding: utf-8 -*-

import numpy as np
import abc


class interface_get_data(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def accel_data(self):
        raise NotImplementedError


@interface_get_data.register
class zurich_data(interface_get_data):
    def __init__(self, subject):
        motions_list = ["posture_changes", "leg_motion", "leg_rotation"]
        self.__subject = subject

        accel_1 = {}
        accel_2 = {}
        accel_3 = {}

        for motion in motions_list:
            accel_1.update({motion: self.__get_accel_data(motion, "1",
                                                          subject)})
            accel_2.update({motion: self.__get_accel_data(motion, "2",
                                                          subject)})
            accel_3.update({motion: self.__get_accel_data(motion, "3",
                                                          subject)})

        self.__accel = {"thight": accel_1, "ankle": accel_2, "wrist": accel_3}
        self.__bioimp = self.__get_bioimp_data("posture_changes",
                                               subject)
        self.accel_dict = {"ankle": 1, "wrist": 2, "thight": 0}

    def accel_data(self):
        return self.__accel

    def __filepath(self, subject, motion_name):
        return "data_zurich/" + subject + "/" + motion_name

    def __get_accel_data(self, motion, accel_id, subject):
        x = np.loadtxt(self.__filepath(subject, motion) + "/X" + accel_id
                       + ".txt").astype(int)
        y = np.loadtxt(self.__filepath(subject, motion) + "/Y" + accel_id
                       + ".txt").astype(int)
        z = np.loadtxt(self.__filepath(subject, motion) + "/Z" + accel_id
                       + ".txt").astype(int)
        data_length = min(x.size, y.size, z.size)

        return np.vstack((x[0:data_length], y[0:data_length],
                          z[0:data_length]))

    def __get_bioimp_data(self, motion, subject):
        freq = np.loadtxt(self.__filepath(subject, motion)
                          + "/F.txt").astype(int)
        phase = np.loadtxt(self.__filepath(subject, motion)
                           + "/P.txt").astype(int)
        resistance = np.loadtxt(self.__filepath(subject, motion)
                                + "/R.txt").astype(int)
        data_length = min(freq.size, phase.size, resistance.size)

        return np.vstack((freq[0:data_length], phase[0:data_length],
                          resistance[0:data_length]))


@interface_get_data.register
class ambica_data(interface_get_data):
    def __init__(self, file_name: str):
        self.__data_type_dict = {"B": "bioimp", "K": "accel", "N": "accel",
                                 "M": "accel", "T": "clock"}
        self.__accel_dict = {"B": "bioimp", "K": "wrist", "N": "thight",
                             "M": "ankle"}
        filepath = "data_ambica/" + file_name
        # self.accel_thight = np.array([])
        # self.accel_wrist = np.array([])
        # self.accel_ankle = np.array([])
        empty_array = np.empty(shape=[0, 3])
        self.__accel_data = {"thight": empty_array, "ankle": empty_array,
                             "wrist": empty_array}
        self.__read_data(filepath)

    def __read_data(self, filepath: str):
        with open(filepath, "r") as file_object:
            data_line = file_object.readlines()

            for data in data_line:
                data_type = data[0]
                self.__save_data(data_type, data)

    def __save_data(self, data_type, data):
        data = data.split(",")
        if self.__data_type_dict[data_type] == "accel":
            self.__save_accel_data(data_type, data)
        elif self.__data_type_dict[data_type] == "bioimp":
            self.__save_bioimp_data(data)

    def __save_accel_data(self, accel_name: str, data):
        accel_data = [int(x) for x in data[2:]]
        self.__accel_data[self.__accel_dict[accel_name]] = \
            np.append(self.__accel_data[self.__accel_dict[accel_name]],
                      np.array([accel_data]), axis=0)

    def __save_bioimp_data(self, data):
        pass

    def accel_data(self):
        return self.__accel_data



if __name__ == "__main__":
    sub1 = zurich_data("s01")
    sub1.accel_data()["ankle"]["posture_changes"]
    
    leg_motion_data = ambica_data("leg_moving.TXT")
    
    accel_data = leg_motion_data.accel_data()
