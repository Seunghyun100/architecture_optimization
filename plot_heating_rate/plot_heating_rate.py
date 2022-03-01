from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import pandas as ps
import os

current_path = os.getcwd()


class HeatingRate:
    def __init__(self):
        pass

    @staticmethod
    def export_csv_data(file_name: str, data: dict):
        data_frame = ps.DataFrame(data)

        file_name = f'{current_path}/data/heating_rate/csv_data/{file_name}'

        if not os.path.exists(f'{file_name}.csv'):
            data_frame.to_csv(f'{file_name}.csv')
            return

        no = 2

        # if already exists same name file
        while os.path.exists(f'{file_name}({no}).csv'):
            no += 1

        # save to csv file
        data_frame.to_csv(f'{file_name}({no}).csv')
        return

    @staticmethod
    def read_csv_data(file_name: str):
        return ps.read_csv(f'{current_path}data/heating_rate/csv_data/{file_name}')

    @staticmethod
    def export_plot_data(data, file_name):

        file_name = f'{current_path}data/heating_rate/plot_data/{file_name}'
        no = 1

        # if already exists same name file
        while os.path.exists(f'{file_name}({no}).png'):
            no += 1

        # save to png file
        data.savefig(f'{file_name}({no}).png')
        return

    def calculate_qbus(self, upto_n: int = 50, upto_t: int = 50, m: int = 1,
                        k_init: float = 0, name: str = "Q-bus_heating_rate"):
        data = {}

        for i in range(upto_n):
            i += 1  # initial n = 1
            # it means k0(initial #phonon) where n = i
            data[i] = [k_init]

        for i in range(upto_n):
            i += 1
            for j in range(upto_t+1):
                if j != 0:
                    pre_k = data[i][j-1]
                    k = (i/(i+m))*(pre_k + 5.2) + 1.1
                    data[i].append(k)

        self.export_csv_data(file_name=name, data=data)
        return data

    def calculate_qbus_ratio(self, upto_n: int = 10, upto_t: int = 50, ratio: tuple = (1, 1),
                             k_init: float = 0, name: str = "Q-bus_heating_rate"):
        data = {}

        for i in range(upto_n):
            i += 1
            i = i*ratio[0]  # initial n = 1 * ratio[0]
            # it means k0(initial #phonon) where n = i
            data[i] = [k_init]

        for i in range(upto_n):
            i += 1
            i = i*ratio[0]
            m = i*ratio[1]
            for j in range(upto_t+1):
                if j != 0:
                    pre_k = data[i][j-1]
                    k = (i/(i+m))*(pre_k + 5.2) + 1.1
                    data[i].append(k)

        self.export_csv_data(file_name=name, data=data)
        return data

    def calculate_closed_system(self, upto_n: int = 50, upto_t: int = 50, m: int = 1,
                        k_init: float = 0, name: str = "closed_system_heating_rate"):
        data = {}

        for i in range(upto_n):
            i += 1  # initial n = 1
            # it means k0(initial #phonon) where n = i
            data[i] = [k_init]

        for i in range(upto_n):
            i += 1
            for j in range(upto_t+1):
                if j != 0:
                    k = (i/(i+m))*(12.4*j - 7.2) + 2
                    data[i].append(k)

        self.export_csv_data(file_name=name, data=data)
        return data

    def calculate_asymm_qccd_grid(self, upto_n: int = 50, upto_t: int = 50, m: int = 1,
                        k_init: float = 0, name: str = "QCCD_heating_rate"):
        total_data = {
            "data_1": {},
            "data_2": {}
        }

        for i in range(upto_n):
            i += 1  # initial n = 1
            # it means k0(initial #phonon) where n = i
            total_data['data_1'][f'{i}'] = [k_init]
            total_data['data_2'][f'{i}'] = [k_init]

        for i in range(upto_n):
            i += 1
            for j in range(upto_t+1):
                if j != 0:
                    if j % 2 == 1:  # if t is odd
                        s_core = 1
                        t_core = 2
                    else:  # if t is even
                        s_core = 2
                        t_core = 1

                    s_core_phonon = (i/(i+m)) * (total_data[f'data_{s_core}'][f'{i}'][j - 1]) + 2
                    t_core_phonon = (m/(i+m)) * (total_data[f'data_{s_core}'][f'{i}'][j - 1]) + \
                                    (total_data[f'data_{t_core}'][f'{i}'][j - 1] + 10.3*(i-2)/(i-1) + 4.1/(i-1))

                    total_data[f'data_{s_core}'][f'{i}'].append(s_core_phonon)
                    total_data[f'data_{t_core}'][f'{i}'].append(t_core_phonon)

        self.export_csv_data(file_name=f'{name}_core1', data=total_data['data_1'])  # core 1
        self.export_csv_data(file_name=f'{name}_core2', data=total_data['data_2'])  # core 2
        return total_data

    def calculate_asymm_qccd_comb(self, upto_n: int = 50, upto_t: int = 50, m: int = 1,
                        k_init: float = 0, name: str = "QCCD_heating_rate"):
        total_data = {
            "data_1": {},
            "data_2": {}
        }

        for i in range(upto_n):
            i += 1  # initial n = 1
            # it means k0(initial #phonon) where n = i
            total_data['data_1'][f'{i}'] = [k_init]
            total_data['data_2'][f'{i}'] = [k_init]

        for i in range(upto_n):
            i += 1
            for j in range(upto_t+1):
                if j != 0:
                    if j % 2 == 1:  # if t is odd
                        s_core = 1
                        t_core = 2
                    else:  # if t is even
                        s_core = 2
                        t_core = 1

                    s_core_phonon = (i/(i+m)) * (total_data[f'data_{s_core}'][f'{i}'][j - 1]) + 2
                    t_core_phonon = (m/(i+m)) * (total_data[f'data_{s_core}'][f'{i}'][j - 1]) + \
                                    (total_data[f'data_{t_core}'][f'{i}'][j - 1] + 10.3)

                    total_data[f'data_{s_core}'][f'{i}'].append(s_core_phonon)
                    total_data[f'data_{t_core}'][f'{i}'].append(t_core_phonon)

        self.export_csv_data(file_name=f'{name}_core1', data=total_data['data_1'])  # core 1
        self.export_csv_data(file_name=f'{name}_core2', data=total_data['data_2'])  # core 2
        return total_data

    def compare_heating_rate(self, qbus_data, qccd_data, name: str = "comparing_Q-bus_vs_QCCD"):
        pros_qccd = {}  # it has boolean value at n and t

        if np.size(qbus_data) == np.size(qccd_data):
            n = len(qbus_data.keys()) - 1  # number of qubit per core
            t = len(qbus_data['1'])  # upto_t + 1

            for i in range(n):
                i += 1
                pros_qccd[f'{i}'] = []
                for j in range(t):
                    if qbus_data[f'{i}'][j] > qccd_data[f'{i}'][j]:
                        pros_qccd[f'{i}'].append(True)
                    else:
                        pros_qccd[f'{i}'].append(False)
            self.export_csv_data(file_name=name, data=pros_qccd)
            return pros_qccd

    def plot_2d(self, x_axis: list, n, y_axis: list = None, name: str = "no_name_2d"):
        if not y_axis:
            y_axis = range(len(x_axis))
        plt.plot(y_axis, x_axis)
        plt.xlabel("#Shuttling")
        plt.xlim(0, 50)
        plt.ylabel("#Phonon")
        plt.title(f"{n} Qubits per core")

        self.export_plot_data(data=plt, file_name=name)
        plt.show()

        return plt

    def plot_3d(self, *data, name: str = "no_name_3d"):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 10),
                                subplot_kw={"projection": "3d"})

        fontlabel = {"fontsize": "large", "color": "gray", "fontweight": "bold"}

        ax.set_xlabel("#qubit per core", fontdict=fontlabel, labelpad=1)
        ax.set_ylabel('#shuttling', fontdict=fontlabel, labelpad=1)
        ax.set_title("#phonon", fontdict=fontlabel)
        ax.view_init(elev=5., azim=130)  # 각도 지정

        cmap_count = 0
        cmap_list = ['inferno', 'Blues', 'Greens']

        for data_i in data:
            # make data frame
            n = len(data_i.keys()) -1
            t = len(data_i) -1

            x = np.array(range(1, n+1))  # X axis is #qubit per core
            y = np.array(range(t+1))  # Y axis is #shuttling
            z = []  # Z axis is #phonon
            x, y = np.meshgrid(x, y)
            x = x.reshape(n*(t+1))
            y = y.reshape(n*(t+1))
            for i in range(len(x)):
                z.append(data_i[f'{x[i]}'][y[i]])

            df = ps.DataFrame({'x': x, 'y': y, 'z': z})
            ax.plot_trisurf(df.x, df.y, df.z, cmap=f'{cmap_list[cmap_count]}', alpha=0.9, linewidth=0.1)
            cmap_count += 1

        # save the figure
        file_name = f'{current_path}/data/heating_rate/plot_data/{name}'
        no = 1
        while os.path.exists(f'{file_name}({no}).png'):  # if already exists same name file
            no += 1
        # save to png file
        fig.savefig(f'{file_name}({no}).png', dpi=600)

        # self.export_plot_data(data=fig, file_name=name)
        plt.show()

    def plot_ratio_3d(self, *data, ratio: tuple = (1, 1), name: str = "no_name_3d"):
        fig, ax = plt.subplots(ncols=1, figsize=(10, 10),
                                subplot_kw={"projection": "3d"})

        fontlabel = {"fontsize": "large", "color": "gray", "fontweight": "bold"}

        ax.set_xlabel("#qubit per core", fontdict=fontlabel, labelpad=1)
        ax.set_ylabel('#shuttling', fontdict=fontlabel, labelpad=1)
        ax.set_title("#phonon with constant ratio", fontdict=fontlabel)
        ax.view_init(elev=20, azim=160)  # 각도 지정

        cmap_count = 0
        cmap_list = ['Blues', 'Greens', 'inferno']

        for data_i in data:
            # make data frame
            n = len(data_i.keys()) -1
            t = len(data_i) -1

            x = np.array(range(1, n+1))  # X axis is #qubit per core
            y = np.array(range(t+1))  # Y axis is #shuttling
            z = []  # Z axis is #phonon
            x, y = np.meshgrid(x, y)
            x = x.reshape(n*(t+1))
            y = y.reshape(n*(t+1))
            for i in range(len(x)):
                z.append(data_i[f'{x[i]*ratio[0]}'][y[i]])

            df = ps.DataFrame({'x': x*ratio[0], 'y': y, 'z': z})
            ax.plot_trisurf(df.x, df.y, df.z, cmap=f'{cmap_list[cmap_count]}', alpha=0.9, linewidth=0.1)
            cmap_count += 1

        # save the figure
        file_name = f'{current_path}/data/heating_rate/plot_data/{name}'
        no = 1
        while os.path.exists(f'{file_name}({no}).png'):  # if already exists same name file
            no += 1
        # save to png file
        fig.savefig(f'{file_name}({no}).png', dpi=600)

        # self.export_plot_data(data=fig, file_name=name)
        plt.show()


    def plot_contour_line(self, data, name):

        fig, ax = plt.subplots(ncols=1, figsize=(10, 10))

        fontlabel = {"fontsize": "large", "color": "gray", "fontweight": "bold"}

        ax.set_xlabel("#qubit per core", fontdict=fontlabel, labelpad=1)
        ax.set_ylabel('#shuttling', fontdict=fontlabel, labelpad=1)


        # make data frame
        n = len(data.keys()) - 1
        t = len(data)
        li = np.zeros(shape=(t, n))
        for i in range(n):
            key = i + 1
            for j in range(t):
                li[j][i] = data[f'{key}'][j]

        plt.ylim(1, 50)  # if 't' insert to 2nd argument, might be too long
        plt.title("Comparing #Phonon")
        ax.imshow(li, cmap="binary")

        # save the figure
        file_name = f'{current_path}/data/heating_rate/plot_data/{name}'
        no = 1
        while os.path.exists(f'{file_name}({no}).png'):  # if already exists same name file
            no += 1
        # save to png file
        fig.savefig(f'{file_name}({no}).png', dpi=600)

        # self.export_plot_data(data=fig, file_name=name)
        plt.show()


if __name__ == '__main__':

    current_path = os.getcwd() + '/../'

    test = HeatingRate()
    # test.calculate_qccd(upto_n=27, upto_t=1000, m=12, name="QCCD_heating_rate_1000_2.25")
    test.calculate_qbus(upto_n=32, upto_t=1000, m=4, name="Q-bus_heating_rate_m4_1000_2.26")
    # test.calculate_qbus_ratio(upto_n=10, upto_t=1000, ratio=(9, 4), name="Q-bus_constant_ratio_heating_rate_1000_2.25")
    # test.calculate_closed_system(upto_n=50, upto_t=1000, m=1, name="closed_sytem_heating_rate_1000_2.25")

    # test_data1 = test.read_csv_data("Q-bus_constant_ratio_heating_rate_1000_2.25.csv")
    test_data1 = test.read_csv_data("Q-bus_heating_rate_m4_1000_2.26.csv")
    # test_data2 = test.read_csv_data("QCCD_heating_rate_1000_2.25_core1.csv")
    # test_data3 = test.read_csv_data("QCCD_heating_rate_1000_2.22_core2.csv")
    # test_data4 = test.read_csv_data("closed_sytem_heating_rate_1000_2.25.csv")

    # test.compare_heating_rate(qccd_data=test_data2, qbus_data=test_data1, name="comparing_n27_m12_2.25")

    # test_data = test.read_csv_data(file_name="comparing_n27_m12_2.25.csv")
    # test.plot_contour_line(data=test_data, name="comparing_n27_m12_2.25")

    # 2D plot
    q_bus_quanta = test_data1["32"]
    # qccd_quanta = test_data2["30"]
    # closed_system = test_data4["30"]
    test.plot_2d(list(q_bus_quanta), n='n(32) m(4)', name="q_bus_n32_m4")
    # test.plot_2d(list(qccd_quanta), n=30, name="qccd_n_30")
    # test.plot_2d(list(closed_system), n=30, name="closed_system_n_30")

    # 3D plot
    # test.plot_3d(test_data1, test_data2, name="Q-bus_vs_QCCD_n27_3d_plot_2.25")
    # test.plot_3d(test_data1, name="Q-bus_heating_rate_1000_2.26")
    # test.plot_ratio_3d(test_data1, ratio=(9, 4), name="Q-bus_constant_ratio_heating_rate_1000_2.25")





