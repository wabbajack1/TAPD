import torch.multiprocessing as _mp
import os

class test_mp():
    def __init__(self, a) -> None:
        print(f"Child process ID: {os.getpid()}, Parent process ID: {os.getppid()}")
        self.a = a
        pass

    def calc(self) -> int:
        print(f"Child process ID: {os.getpid()}, Parent process ID: {os.getppid()}", self.a * self.a)
        return self.a * self.a
    
if __name__ == "__main__":
    m = test_mp(10)

    mp = _mp.get_context("spawn")

    processes = []
    for index in range(10):
        if index == 0:
            process = mp.Process(target=m.calc)
        else:
            process = mp.Process(target=m.calc)
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    m.calc()

    pass