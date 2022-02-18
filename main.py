from plot_heating_rate import plot_heating_rate
from simulation_execution_time import simulation_execution_time


def heating_rate():
    test = plot_heating_rate.HeatingRate()
    test.calculate_qccd(upto_n = 50, upto_t = 100, m = 5, name="QCCD_heating_rate_m5_100")
    test.calculate_qbus(upto_n=50, upto_t=100, m=5, name="Q-bus_heating_rate_m5_100")

    # test_data1 = test.read_csv_data("QCCD_heating_rate_m5_100_core1.csv")
    # test_data2 = test.read_csv_data("QCCD_heating_rate_m5_100_core2.csv")
    # test_data3 = test.read_csv_data("Q-bus_heating_rate_m5_100.csv")
    #
    # test.compare_heating_rate(qccd_data=test_data2, qbus_data=test_data3)

    test_data = test.read_csv_data(file_name="comparing_Q-bus_vs_QCCD(8).csv")
    test.plot_contour_line(data=test_data, name="test")

    # test.plot_2d(list(x), name="test")

    # test.plot_3d(test_data1, test_data2, test_data3, name="Q-bus_QCCD_3d_plot")


def sim_execution():
    pass


if __name__ == '__main__':
    heating_rate()
    sim_execution()
