import numpy as np

"""
This below part is independent of the hardware types
"""

class Circuit:
    """
    num_qubits(int) : number of qubits. It should be integer multiple of num_cores
    num_cores(int) : number of cores. They have same number of qubits.
                    The capacity of each core is num_qubits divided by num_cores.
                    We would evaluate 6 cores architecture. So default value is 6.
    hardware(str) : Hardware configuration. i.e., none, qbus, qccd_comb, qccd_grid
    is_symmetric(bool) : whether it is symmetric of inter communication.
    """
    def __init__(self, num_qubits: int, num_cores: int = 6, hardware: str = 'none', is_symmetric: bool = False):
        if num_qubits%num_cores != 0:
            raise print("Number of qubits is not integer multiple of number of cores.")

        self.hardware = hardware
        self.is_symmetric = is_symmetric
        self.num_qubits = num_qubits
        self.capacity = int(num_qubits/num_cores)

        topology = self.init_circuit(num_qubits=num_qubits, num_cores=num_cores)
        self.topology = topology
        self.qubit_list = topology['qubits']
        self.core_list = topology['cores']

    def init_circuit(self, num_qubits, num_cores):
        topology = {
            'num_qubits': num_qubits,
            'num_cores': num_cores,
            'cores': [],
            'qubits': []
        }
        capacity = self.capacity

        for i in range(num_cores):
            topology['cores'].append(Core(capacity=capacity, address=i))
            topology['qubits'].append([])

        for i in range(num_cores):
            for k in range(capacity):
                qubit_index = i*capacity + k
                qubit = Qubit(core_address=i, index=qubit_index)
                topology['qubits'][i].append(qubit)
                topology['cores'][i].qubits.append(qubit)

        return topology

    def operate(self, operation_name: str, *qubits):
        operation = Operation(operation_name=operation_name, *qubits)
        for i in qubits:
            self.qubit_list[i].oeprations.append(operation)
        return

    def calculate_execution_time(self):
        execution_time_list = []
        # update operation time list
        for qubit in self.qubit_list:
            qubit.build_auxiliary_time_list()
            if self.hardware != 'none':  # TODO : WIP
                self.insert_inter_comm(qubit)

        trigger = True
        while trigger:
            for qubit in self.qubit_list:
                self.merge_int(qubit)
                if not self.check_done_qubit(qubit):
                    operation = qubit.auxiliary_time_list[1][2]
                    control_qubit = operation.qubit
                    target_qubit = operation.target_qubit

                    if self.check_first_two(qubit=control_qubit, operation=operation) \
                            and self.check_first_two(qubit=target_qubit, operation=operation):
                        self.synchronize_timing(control_qubit, target_qubit)

            # Break the while loop if finish to synchronize
            for qubit in self.qubit_list:
                if not self.check_done_qubit(qubit):
                    trigger = True
                    break
                trigger = False

            if not trigger:
                for qubit in self.qubit_list:
                    execution_time_list.append(qubit.auxiliary_time_list[0])

        return execution_time_list

    @staticmethod
    def check_done_qubit(qubit):
        done = False
        if len(qubit.auxiliary_time_list) == 1:
            done = True
        return done

    def merge_int(self, qubit):
        if not self.check_done_qubit(qubit):
            while type(qubit.auxiliary_time_list[1]) == int and not self.check_done_qubit(qubit):
                qubit.auxiliary_time_list[0] += qubit.auxiliary_time_list.pop(1)
        return qubit

    def check_first_two(self, qubit, operation):
        is_first_two = False
        if not self.check_done_qubit(qubit):
            if qubit.auxiliary_time_list[1][2] == operation:
                is_first_two = True
        return is_first_two

    @staticmethod
    def synchronize_timing(control_qubit, target_qubit):
        synchronized_timing = max(control_qubit.auxiliary_time_list[0], target_qubit.auxiliary_time_list[0])
        operation_time = control_qubit.auxiliary_time_list[1][2].time
        synchronized_timing += operation_time

        control_qubit.auxiliary_time_list[0] = synchronized_timing
        target_qubit.auxiliary_time_list[0] = synchronized_timing

        control_qubit.auxiliary_time_list.pop(1)
        target_qubit.auxiliary_time_list.pop(1)

        return control_qubit, target_qubit

    # TODO : WIP
    def insert_inter_comm(self, qubit):
        auxiliary_time_list = qubit.auxiliary_time_list
        for i in range(len(auxiliary_time_list)):
            if type(auxiliary_time_list[i]) == list:
                operation = auxiliary_time_list[i]
                if operation[0] == 'inter':
                    inter_comm = InterComm(hardware=self.hardware, original_oepration=operation[2],
                                           is_symmetric=self.is_symmetric)





class Qubit:
    """
    core_address(int) : Location of core. It would be used to configure which core it belongs to.
    """
    def __init__(self, core_address: int, index: int):
        self.index = index
        self.operations = []
        self.time_list = []
        self.auxiliary_time_list = []  # clustering single gate to synchronize 2-qubit gate timing.
        self.core_address = core_address

    def arrange_time_list(self):
        time_list = []
        for i in range(len(self.operations)):
            operation = self.operations[i]
            if operation.type == 'one':
                time_list.append(['one', operation.time, operation])
            if operation.type == 'two':
                if operation.is_inter_comm:
                    time_list.append(['inter', operation.time, operation])
                else:
                    time_list.append(['intra', operation.time, operation])
            if operation.type == 'init':
                time_list.append(['init', operation.time, operation])
            if operation.type == 'detection':
                time_list.append(['detection', operation.time, operation])

        self.time_list = time_list
        return self.time_list

    def build_auxiliary_time_list(self):
        auxiliary_time_list = [0]
        time_list = self.arrange_time_list()

        for i in time_list:
            if i[0] == 'intra' or 'inter':
                auxiliary_time_list.append(i)
            else:
                if type(auxiliary_time_list[-1]) == list:
                    auxiliary_time_list.append(i[1])
                else:
                    auxiliary_time_list[-1] += i[1]

        self.auxiliary_time_list = auxiliary_time_list
        return self.auxiliary_time_list

    def commute_operation(self, operation1: int, operation2: int):  # parameter is index of operation in list
        li = self.operations
        li[operation1], li[operation2] = li[operation2], li[operation1]
        self.operations = li
        return self.operations


class Core:
    """
    location(int) : Location of core. It determines topology like shuttling path.
    capacity(int) : Capacity of core
    """
    def __init__(self, capacity: int, address: int):
        self.qubits = []
        self.address = address
        self.capacity = capacity


class Operation:
    """
    operation_name(str) : name of operation.
        i.e., x, y, h, cx, detection, init
    *qubits(object) : operated qubit. we assumed not used 3-more qubits gate.
                        If applied 2-qubit gate, qubits[0] is controlled qubit and qubits[1] is target qubit.
    """
    def __init__(self, operation_name: str, *qubits):
        self.name = operation_name
        self.is_inter_comm = self.check_is_inter_comm(qubits[0], qubits[1])
        self.type = self.check_type(operation_name=operation_name)  # i.e., one, two(intra or inter), init, detection
        self.commute_list = self.check_comm_list(operation_name)
        self.qubit = qubits[0]
        self.target_qubit = qubits[1]
        self.time = self.calculate_time()

    # TODO : build a commutable list of each gate
    @staticmethod
    def check_comm_list(operation_name):
        commute_dict = {
            'i': ['i', 'x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'rxx', 'ryy', 'rzz', 'swap'],
            'x': ['i', 'x', 'y', 'z', 'rx', 'cx_t', 'cy_t', 'cz', 'rxx'],
            'y': ['i', 'x', 'y', 'z', 'ry', 'cx_t', 'cy_t', 'cz', 'ryy'],
            'z': ['i', 'x', 'y', 'z', 'rz', 'cx', 'cy', 'cz', 'rzz'],
            'h': [],
            's': [],
            't': [],
            'rx': [],
            'ry': [],
            'rz': [],
            'cx': [],
            'cy': [],
            'cz': [],
            'rxx': [],
            'ryy': [],
            'rzz': [],
            'swap': []
        }
        return commute_dict[operation_name]

    @staticmethod
    def check_is_inter_comm(c_qubit, t_qubit):
        is_inter_comm = False
        if not c_qubit.core_address == t_qubit.core_address:
            is_inter_comm = True
        return is_inter_comm

    def check_type(self,operation_name):
        operation_type = str
        gate_list = ['i', 'x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'rxx', 'ryy', 'rzz', 'swap']
        one = ['x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz']
        two = ['cx', 'cy', 'cz', 'rxx', 'ryy', 'rzz', 'swap']
        init = 'init'
        detection = 'detection'

        if operation_name in one:
            operation_type = 'one'

        if operation_name in two:
            operation_type = 'two'

        if operation_name == init:
            operation_type = init

        if operation_name == detection:
            operation_type = detection

        return operation_type

    def calculate_time(self):
        # time unit is micro second. 'init' might not be used.
        time = {
            'one': 5,
            'two': 40,
            'init': int,
            'detection': 180,
        }

        return time[f'{self.type}']


"""
    The below part depends on hardware type. 
"""


# TODO : WIP
class InterComm:
    def __init__(self, hardware: str, original_oepration, is_symmetric: bool = False):
        self.hardware = hardware
        self.original_operation = original_oepration
        self.is_symmetric = is_symmetric
        self.operation_list = self.build_inter_comm()  # It is procedure list of inter communication.
        self.time_list = self.calculate_time_list()  # It is to calculate inter communication time(execution time).

    def build_inter_comm(self):
        # We assume every inter communication is CNOT gate where Q-bus.
        operation_list = []
        if self.hardware == 'qbus':
            pass
        elif self.hardware == 'qccd_comb':
            pass
        elif self.hardware == 'qccd_grid':
            pass
        return operation_list

    def calculate_time_list(self):
        time_list = []
        return time_list


class Scheduler:
    def __init__(self):
        pass


class Mapper:
    def __init__(self):
        pass


class Simulator:
    def __init__(self):
        pass

    # TODO : Calculate circuit execution time.
    #  To calculate circuit execution time, we should sum circuit.qubit.time (list)
    #  But if there are two qubit gate, we would add delay to synchronize operation timing.
    def execution(self, circuit):
        execution_time = 0


        return execution_time


if __name__=="__main__":
    pass
