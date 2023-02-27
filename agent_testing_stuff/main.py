from ProgDenseBlock import ProgDenseBlock
import torch

def main():
    b = ProgDenseBlock(4, 5, 0)
    x = torch.tensor([[1.0, 2.5, 4.4, 5.2], [0.8, 2.3, 1.9, 7.3], [3.6, 3.3, 1.7, 8.1]])
    print(b.runActivation(b.runBlock(x)))

if __name__ == "__main__":
    main()