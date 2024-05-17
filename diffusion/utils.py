import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m: nn.Module):
    """Initialize the weights of the given module."""

    def truncated_normal_init(t: torch.Tensor, mean=0.0, std=0.01):
        """Initialize a tensor with a truncated normal distribution."""
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(
                cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t
            )
        return t

    if type(m) == nn.Linear:
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding module."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the sinusoidal positional embedding of the input."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extracts elements from `a` at indices given by `t` and reshapes to match `x_shape`."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s=0.008, dtype=torch.float32) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(
    timesteps: int, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32
) -> torch.Tensor:
    """Generates a linear beta schedule from `beta_start` to `beta_end` over `timesteps`."""
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps: int, dtype=torch.float32) -> torch.Tensor:
    """Generates a vp beta schedule over `timesteps`."""
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class WeightedLoss(nn.Module):
    """A loss module that applies weights to the loss values."""

    def forward(
        self, pred: torch.Tensor, targ: torch.Tensor, weights: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the weighted loss.
        pred, targ : tensor [ batch_size x action_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss):
    """A loss module that applies weights to the L1 loss values."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    """A loss module that applies weights to the L2 loss values."""

    def _loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}


class EMA:
    """
    Empirical Moving Average (EMA) class.
    """

    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module):
        """Update the moving average of the model parameters."""
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Update the moving average of a tensor."""
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def print_banner(s, separator="-", num_star=60):
    """
    Prints a banner with a message surrounded by separator lines.

    Args:
        s (str): The message to print.
        separator (str, optional): The character to use for the separator lines. Defaults to "-".
        num_star (int, optional): The number of separator characters to use for the separator lines. Defaults to 60.
    """
    # Print the top separator line
    print(separator * num_star, flush=True)
    # Print the message
    print(s, flush=True)
    # Print the bottom separator line
    print(separator * num_star, flush=True)


class Progress:
    def __init__(
        self,
        total,
        name="Progress",
        ncol=3,
        max_length=20,
        indent=0,
        line_width=100,
        speed_update_freq=100,
    ):
        self.total = total
        self.name = name
        self.ncol = ncol
        self.max_length = max_length
        self.indent = indent
        self.line_width = line_width
        self._speed_update_freq = speed_update_freq

        self._step = 0
        self._prev_line = "\033[F"
        self._clear_line = " " * self.line_width

        self._pbar_size = self.ncol * self.max_length
        self._complete_pbar = "#" * self._pbar_size
        self._incomplete_pbar = " " * self._pbar_size

        self.lines = [""]
        self.fraction = "{} / {}".format(0, self.total)

        self.resume()

    def update(self, description, n=1):
        self._step += n
        if self._step % self._speed_update_freq == 0:
            self._time0 = time.time()
            self._step0 = self._step
        self.set_description(description)

    def resume(self):
        self._skip_lines = 1
        print("\n", end="")
        self._time0 = time.time()
        self._step0 = self._step

    def pause(self):
        self._clear()
        self._skip_lines = 1

    def set_description(self, params=[]):
        if type(params) == dict:
            params = sorted([(key, val) for key, val in params.items()])

        ############
        # Position #
        ############
        self._clear()

        ###########
        # Percent #
        ###########
        percent, fraction = self._format_percent(self._step, self.total)
        self.fraction = fraction

        #########
        # Speed #
        #########
        speed = self._format_speed(self._step)

        ##########
        # Params #
        ##########
        num_params = len(params)
        nrow = math.ceil(num_params / self.ncol)
        params_split = self._chunk(params, self.ncol)
        params_string, lines = self._format(params_split)
        self.lines = lines

        description = "{} | {}{}".format(percent, speed, params_string)
        print(description)
        self._skip_lines = nrow + 1

    def append_description(self, descr):
        self.lines.append(descr)

    def _clear(self):
        position = self._prev_line * self._skip_lines
        empty = "\n".join([self._clear_line for _ in range(self._skip_lines)])
        print(position, end="")
        print(empty)
        print(position, end="")

    def _format_percent(self, n, total):
        if total:
            percent = n / float(total)

            complete_entries = int(percent * self._pbar_size)
            incomplete_entries = self._pbar_size - complete_entries

            pbar = (
                self._complete_pbar[:complete_entries]
                + self._incomplete_pbar[:incomplete_entries]
            )
            fraction = "{} / {}".format(n, total)
            string = "{} [{}] {:3d}%".format(fraction, pbar, int(percent * 100))
        else:
            fraction = "{}".format(n)
            string = "{} iterations".format(n)
        return string, fraction

    def _format_speed(self, n):
        num_steps = n - self._step0
        t = time.time() - self._time0
        speed = num_steps / t
        string = "{:.1f} Hz".format(speed)
        if num_steps > 0:
            self._speed = string
        return string

    def _chunk(self, l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]

    def _format(self, chunks):
        lines = [self._format_chunk(chunk) for chunk in chunks]
        lines.insert(0, "")
        padding = "\n" + " " * self.indent
        string = padding.join(lines)
        return string, lines

    def _format_chunk(self, chunk):
        line = " | ".join([self._format_param(param) for param in chunk])
        return line

    def _format_param(self, param):
        k, v = param
        return "{} : {}".format(k, v)[: self.max_length]

    def stamp(self):
        if self.lines != [""]:
            params = " | ".join(self.lines)
            string = "[ {} ] {}{} | {}".format(
                self.name, self.fraction, params, self._speed
            )
            self._clear()
            print(string, end="\n")
            self._skip_lines = 1
        else:
            self._clear()
            self._skip_lines = 0

    def close(self):
        self.pause()


class Silent:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args: None


class EarlyStopping(object):
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                return True
        else:
            self.counter = 0
        return False


def mean_flat(tensor):
    """
    Compute the mean over all non-batch dimensions of a tensor.

    This function takes a tensor as input and computes the mean over all dimensions except the first one (batch dimension).
    It is assumed that the first dimension of the tensor is the batch dimension.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The mean of the input tensor over all non-batch dimensions.
    """
    # Create a list of all non-batch dimensions
    non_batch_dims = list(range(1, len(tensor.shape)))
    # Compute the mean over these dimensions
    return tensor.mean(dim=non_batch_dims)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.

    Args:
        mean1 (torch.Tensor or float): The mean of the first gaussian.
        logvar1 (torch.Tensor or float): The log variance of the first gaussian.
        mean2 (torch.Tensor or float): The mean of the second gaussian.
        logvar2 (torch.Tensor or float): The log variance of the second gaussian.

    Returns:
        torch.Tensor: The KL divergence between the two gaussians.
    """
    # Check if at least one argument is a Tensor
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Ensure logvar1 and logvar2 are Tensors
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # Compute the KL divergence
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    Compute a fast approximation of the cumulative distribution function (CDF) of the standard normal distribution.

    This function uses a specific formula based on the tanh function and the cube function to compute an approximation of the CDF.
    This approximation is fast and does not sacrifice too much accuracy.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The approximated CDF of the standard normal distribution.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    Args:
        x (torch.Tensor): The target images. It is assumed that this was uint8 values,
        rescaled to the range [-1, 1].
        means (torch.Tensor): The Gaussian mean Tensor.
        log_scales (torch.Tensor): The Gaussian log stddev Tensor.

    Returns:
        torch.Tensor: A tensor like x of log probabilities (in nats).
    """
    # Ensure the shapes of x, means, and log_scales are the same
    assert x.shape == means.shape == log_scales.shape

    # Compute the centered x
    centered_x = x - means

    # Compute the inverse standard deviation
    inv_stdv = torch.exp(-log_scales)

    # Compute the CDF of the upper bound of x
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    # Compute the CDF of the lower bound of x
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    # Compute the log CDF of the upper bound of x, and clamp the value to avoid numerical issues
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))

    # Compute the log (1 - CDF) of the lower bound of x, and clamp the value to avoid numerical issues
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    # Compute the difference between the CDFs of the upper and lower bounds of x, and clamp the value to avoid numerical issues
    cdf_delta = cdf_plus - cdf_min

    # Compute the log probabilities
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )

    # Ensure the shape of log_probs is the same as x
    assert log_probs.shape == x.shape

    return log_probs
