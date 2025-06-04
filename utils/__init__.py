import torch


dtypes = {
    "float32": torch.float32,
    "float16": torch.float16,
    "double": torch.double,
    "float": torch.float,
    "half": torch.float16,
}
loss_functions = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
}
