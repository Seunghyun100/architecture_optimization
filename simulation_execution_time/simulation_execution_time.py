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
    """
    def __init__(self, num_qubits: int, num_cores: int = 6):
        if num_qubits%num_cores != 0:
            raise print("Number of qubits is not integer multiple of number of cores.")

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
        time = 0
        # update operation time list
        for i in self.qubit_list:
            i.arrange_time_list()
            i.calculate_auxiliary_time_list()

        trigger = True
        while trigger:
            for i in self.qubit_list:
                if self.check_first_two(i):
                    control_qubit = i.auxiliary_time_list[0][2].qubit
                    target_qubit = i.auxiliary_time_list[0][2].target_qubit
                    if self.check_first_two(control_qubit) and self.check_first_two(target_qubit):
                        self.synchronize_timing()

            # Break the while loop if finish to synchronize
            for i in self.qubit_list:
                if not self.check_done_qubit(i):
                    break
                trigger = False

        return time

    @staticmethod
    def check_done_qubit(qubit):
        done = False
        if len(qubit.auxiliary_time_list) == 0:
            qubit.auxiliary_time_list.append(0)
            done = True
        if len(qubit.auxiliary_time_list) == 1:
            if type(qubit.auxiliary_time_list[0]) == int:
                done = True
        return done

    @staticmethod
    def check_first_two(qubit):
        is_first_two = False
        if qubit.auxiliary_time_list[0] == list:
            is_first_two = True
        elif qubit.auxiliary_time_list[1] == list:
            is_first_two = True
        return is_first_two

    def synchronize_timing(self, operation):
        pass

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

    def calculate_auxiliary_time_list(self):
        auxiliary_time_list = []
        time_list = self.arrange_time_list()
        index = 0

        for i in time_list:
            if i[0] == 'intra' or 'inter':
                auxiliary_time_list.append(i)
            else:
                if len(auxiliary_time_list) == 0:
                    auxiliary_time_list.append(i[1])
                elif type(auxiliary_time_list[len(auxiliary_time_list)-1]) == list:
                    auxiliary_time_list.append(i[1])
                else:
                    auxiliary_time_list[len(auxiliary_time_list)-1] += i[1]

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
        # self.is_delay = self.check_is_delay()

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
        # time unit is micro second. 'init' might not be used. 'inter' depends on hardware type.
        time = {
            'one': 5,
            'two': 40,
            'init': int,
            'detection': 180,
        }

        # TODO : create a function about time of inter comm considering hardware type.
        if self.is_inter_comm:
            time['two'] = int

        return time[f'{self.type}']

    # It is to synchronize gate execution for two qubit gate
    # def check_is_delay(self):
    #     delay = False
    #     if self.type == "two":
    #         delay = True
    #     return delay


"""
    The below part depends on hardware type. 
"""


class InterComm:
    """
    is_path(bool) : Determine whether it is a path or a junction
    """
    def __init__(self, is_path: bool):
        self.is_path = is_path  # If True means 'path' or False means 'junction'


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
