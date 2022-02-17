
class Circuit:
    """
    num_qubits(int) : number of qubits.
    capacity(int) : capacity of a core. We would evaluate 6 cores architecture.
                    The capacity is about each core. And they have same number of qubits.
    """
    def __init__(self, num_qubits, capacity):
        self.num_qubits = num_qubits
        self.capacity = capacity
        self.qubit_list = []
        self.core_list = []


class Qubit:
    """
    core_address(int) : Location of core. It would be used to configure which core it belongs to.
    """
    def __init__(self, core_address:int):
        self.id = int
        self.circuit = []
        self.time = 0
        self.core = core_address


class Core:
    """
    location(int) : Location of core. It determines topology like shuttling path.
    capacity(int) : Capacity of core
    """
    def __init__(self, capacity:int, location:int):
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
    def __init__(self, operation_name:str, *qubits):
        self.name = operation_name
        self.type = str
        self.commute_list = self.checkCommList(operation_name)
        self.is_inter_comm = self.isInterComm(qubits[0], qubits[1])
        self.qubit = qubits[0]
        self.target_qubit = qubits[1]

    @staticmethod
    def checkCommList(operation_name):
        commute_dict = {
            'i' : [],
            'x' : [],
            'y' : [],
            'z' : [],
            'h' : [],
            's' : [],
            't' : [],
            'rx' : [],
            'ry' : [],
            'rz' : [],
            'cx' : [],
            'cy' : [],
            'cz' : [],
            'rxx' : [],
            'ryy' : [],
            'rzz' : [],
            'swap' : []
        }
        return commute_dict[operation_name]

    def isInterComm(self, c_qubit, t_qubit):
        is_inter_comm =False
        if not c_qubit.core == t_qubit.core:
            is_inter_comm = True
        return is_inter_comm


class Shuttling:
    """
    is_path(bool) : Determine whether it is a path or a junction
    """
    def __init__(self, is_path:bool):
        self.is_path = is_path  # If True means 'path' or False means 'junction'



if __name__=="__main__":
    pass
