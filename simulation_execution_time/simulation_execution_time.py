

class Qubit:
    """
    core_address(int) : Location of core. It would be used to configure which core it belongs to.
    """
    def __init__(self, core_address:int):
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
        i.e., x, y, h, cnot, detection, init

    """
    def __init__(self, operation_name:str):
        self.name = operation_name
        self.is_inter_comm = False
        self.type = str
        self.commute_list = self.detCommList(operation_name)

    def detCommList(self, operation_name):
        commute_list = []

        return commute_list

    def isInterComm(self,):


class Shuttling:
    """
    is_path(bool) : Determine whether it is a path or a junction
    """
    def __init__(self, is_path:bool):
        self.is_path = is_path  # If True means 'path' or False means 'junction'
        self.


if __name__=="__main__":
    pass
