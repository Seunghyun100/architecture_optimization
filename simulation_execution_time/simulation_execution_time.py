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
            topology['cores'].append(Core(capacity=capacity, location=i))
            topology['qubits'].append([])

        for i in range(num_cores):
            for k in range(capacity):
                qubit_index = i*capacity + k
                qubit = Qubit(core_address=i, index=qubit_index)
                topology['qubits'][i].append(qubit)
                topology['cores'][i].qubits.append(qubit)

        return topology

    # TODO : create a function to calculate time
    def operate(self, operation_name: str, *qubits):
        operation = Operation(operation_name=operation_name, *qubits)
        for i in qubits:
            self.qubit_list[i] = operation

        if operation.type != 'two' or 'inter':
            pass

        return


class Qubit:
    """
    core_address(int) : Location of core. It would be used to configure which core it belongs to.
    """
    def __init__(self, core_address: int, index: int):
        self.index = index
        self.operations = []
        self.time = 0
        self.core = core_address

    def calculate_time(self):
        time = 0

        return time

    def synchronize_time(self):
        pass



class Core:
    """
    location(int) : Location of core. It determines topology like shuttling path.
    capacity(int) : Capacity of core
    """
    def __init__(self, capacity: int, location: int):
        self.qubits = []
        self.location = location
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
        self.type = self.check_type(operation_name=operation_name)  # i.e., one, two, init, detection, inter
        self.commute_list = self.check_comm_list(operation_name)
        self.qubit = qubits[0]
        self.target_qubit = qubits[1]
        self.time = self.check_time()
        self.delay = self.check_is_delay()

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
        if not c_qubit.core == t_qubit.core:
            is_inter_comm = True
        return is_inter_comm

    def check_type(self,operation_name):
        operation_type = str
        gate_list = ['i', 'x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz', 'cx', 'cy', 'cz', 'rxx', 'ryy', 'rzz', 'swap']
        one = ['x', 'y', 'z', 'h', 's', 't', 'rx', 'ry', 'rz']
        two = ['cx', 'cy', 'cz', 'rxx', 'ryy', 'rzz', 'swap']
        init = 'init'
        detection = 'detection'
        inter = 'inter'

        if operation_name in one:
            operation_type = 'one'

        if operation_name in two:
            if self.is_inter_comm:
                operation_type = inter
            else:
                operation_type = 'two'

        if operation_name == init:
            operation_type = init

        if operation_name == detection:
            operation_type = detection

        return operation_type

    def check_time(self):
        # time unit is micro second. 'init' might not be used. 'inter' depends on hardware type.
        time = {
            'one': 5,
            'two': 40,
            'init': int,
            'detection': 180,
            'inter': int
        }  # TODO : create a function about time of inter comm considering hardware type.

        return time[f'{self.type}']

    # TODO : create a function considering Scheduler.
    def check_is_delay(self):
        delay = 0

        return delay


"""
    The below part depends on hardware type. 
"""


class Shuttling:
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




if __name__=="__main__":
    pass
