import torch
import numpy as np
import os


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


"""
GPU wrappers
"""

_use_gpu = False
device = None
_gpu_id = 0


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    if not _use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = None
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")
    device = torch.device("cuda" if _use_gpu else "cpu")
    # device = torch.device("cuda:" + str(2) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(np_array, requires_grad=False):
    # tensor_array = torch.from_numpy(np_array).float().to(device)
    tensor_array = torch.from_numpy(np_array.astype(np.float32)).to(
        device
    )  # float() creates a new tensor, making it much more slow
    tensor_array.requires_grad_(requires_grad=requires_grad)
    return tensor_array


def get_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to("cpu").detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def tf_like_gather(x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
    # perform gather like tensorflow
    return torch.cat([x[j][None] for j in i])


def tf_like_gather_np(x: np.ndarray, i: np.ndarray) -> np.ndarray:
    return np.concatenate([x[j][None] for j in i])


def print_grad_hook(grad):
    # for debug
    print(grad)
