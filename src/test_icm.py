import utils as utils
import commons.model as model
from pathlib import Path
import datetime
import torch

if __name__ == "__main__":
    m = model.ICM(1)
    kb_column = model.KB_Module("cpu")
    active_column = model.Active_Module("cpu", True)
    prog = model.ProgressiveNet(kb_column, active_column)

    for i in prog.named_parameters():
        print(i[0])

    state = torch.randn(1, 1, 84, 84)
    action = torch.randint(0, 17, (18,))
    predicted_action, predicted_next_state, phi_next_state = m(state, state, 2)
    print(predicted_action.shape, predicted_next_state.shape, phi_next_state.shape)
    pass