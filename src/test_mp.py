import torch.multiprocessing as _mp
import os
import random
from ctypes import c_int, addressof


class test_mp():
    def __init__(self) -> None:
        print(f"Child process ID: {os.getpid()}, Parent process ID: {os.getppid()}")
        self.same = 10
        pass

    def calc(self, index) -> int:
        random.seed(10+index)
        self.a = random.randint(1, 1000)
        print(f"Index {index}, Child process ID: {os.getpid()}, Parent process ID: {os.getppid()}, memory address {addressof(c_int(self.same))}", self.a, self.a * self.a)
        return self.a * self.a
    
if __name__ == "__main__":
    m = test_mp()

    mp = _mp.get_context("spawn")

    processes = []
    for index in range(10):
        if index == 0:
            process = mp.Process(target=m.calc, args=[index])
        else:
            process = mp.Process(target=m.calc, args=[index])
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    m.calc(0)

    pass