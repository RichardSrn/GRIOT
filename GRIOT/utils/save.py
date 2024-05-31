import os
import numpy as np
import torch

def save_steps(matrix, path, graph_name, *args, **kwargs):
    path = os.path.join(path, "imp_steps")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, graph_name)
    os.makedirs(path, exist_ok=True)
    file_name = ""
    for k in args:
        file_name += f"{k}_"
    for k, v in kwargs.items():
        try:
            file_name += f"{k}{float(v):.2f}_"
        except ValueError:
            file_name += f"{k}{v}_"
    if type(matrix) == dict:
        save_path = []
        for i, step in matrix.items():
            save_path.append(os.path.join(path, file_name + f"_step={i}.npy"))
            if torch.is_tensor(step):
                step = step.cpu().detach().numpy()
            np.save(save_path[-1], step)
    else: # np.ndarray or torch.Tensor
        save_path = os.path.join(path, file_name + f".npy")
        if torch.is_tensor(matrix):
            matrix = matrix.cpu().detach().numpy()
        np.save(save_path, matrix)
    return save_path
