"""
Based on rllab's logger.

https://github.com/rll/rllab
"""
from enum import Enum
from contextlib import contextmanager
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil.tz
import csv
import joblib
import json
import errno

try:
    import wandb

    _wandb_installed = True
except ImportError:
    _wandb_installed = False

from algorithms.rlkit.core.tabulate import tabulate
from algorithms.rlkit.launchers import config
from torch.utils.tensorboard import SummaryWriter


def load_from_file(algorithm, load_replay_buffer, load_model, load_path=None):
    if (load_path is not None) and (len(load_path) > 0):
        load_extra_data_path = load_path + "/extra_data.pkl"
        load_model_path = load_path + "/params.pkl"

        extra_data = joblib.load(load_extra_data_path)
        model = joblib.load(load_model_path)

        if load_replay_buffer:
            print("LOAD BUFFER from {}".format(load_path))
            algorithm.replay_buffer = extra_data["replay_buffer"]
        if load_model:
            print("LOAD MODELS from {}".format(load_path))
            algorithm.load_snapshot(model)

        algorithm.set_steps(extra_data)
        epoch = extra_data["epoch"] + 1

    return algorithm, epoch


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


_prefixes = []
_prefix_str = ""

_tabular_prefixes = []
_tabular_prefix_str = ""

_tabular = []

_text_outputs = []
_tabular_outputs = []

_text_fds = {}
_tabular_fds = {}
_tabular_header_written = set()

_snapshot_dir = None
_snapshot_mode = "all"
_snapshot_gap = 1

_log_tabular_only = False
_header_printed = False
_log_tboard = True
_log_wandb = True
_step_key = "Epoch"

_summary_writer = None


def _add_output(file_name, arr, fds, mode="a"):
    if file_name not in arr:
        mkdir_p(os.path.dirname(file_name))
        arr.append(file_name)
        fds[file_name] = open(file_name, mode)


def _remove_output(file_name, arr, fds):
    if file_name in arr:
        fds[file_name].close()
        del fds[file_name]
        arr.remove(file_name)


def push_prefix(prefix):
    _prefixes.append(prefix)
    global _prefix_str
    _prefix_str = "".join(_prefixes)


def add_text_output(file_name):
    _add_output(file_name, _text_outputs, _text_fds, mode="a")


def remove_text_output(file_name):
    _remove_output(file_name, _text_outputs, _text_fds)


def add_tabular_output(file_name):
    _add_output(file_name, _tabular_outputs, _tabular_fds, mode="w")


def remove_tabular_output(file_name):
    if _tabular_fds[file_name] in _tabular_header_written:
        _tabular_header_written.remove(_tabular_fds[file_name])
    _remove_output(file_name, _tabular_outputs, _tabular_fds)


def set_tboard(dir_name, name="tboard"):
    log_path = osp.join(dir_name, name)
    global _summary_writer
    _summary_writer = SummaryWriter(log_path)


def set_wandb(dir_name, variant):
    # if variant["env_specs"]["disc_ddpm"]:
    #     name = variant["exp_name"] + '-' + str(variant["disc_ddpm_n_timesteps"]) + '-' \
    #            + variant["disc_ddpm_beta_schedule"] + '-' + str(variant["seed"]) + \
    #            '-grad_' + str(variant["adv_irl_params"]["grad_pen_weight"]) + '-' + str(
    #         variant["sac_params"]["reward_scale"])
    # else:
    #     name = variant["exp_name"] + '-grad_' + str(variant["adv_irl_params"]["use_grad_pen"])
    # wandb.init(
    #     project=config.PROJECT_NAME,
    #     dir=dir_name,
    #     config=variant,
    #     name=name,
    # )

    wandb.init(
        project=config.PROJECT_NAME,
        dir=dir_name,
        config=variant,
        name=variant["exp_name"],
    )


def set_snapshot_dir(dir_name, variant, log_tboard=True, log_wandb=True):
    global _snapshot_dir, _log_tboard, _log_wandb
    _snapshot_dir = dir_name
    _log_tboard = log_tboard
    if _log_wandb and not _wandb_installed:
        print("\nWandb not installed. Not logging to wandb.")
        print(
            "Please install wandb with 'pip install wandb', or turn off wandb logging\n"
        )
        _log_wandb = False
    else:
        _log_wandb = log_wandb
    if _log_tboard:
        set_tboard(dir_name)
    if _log_wandb:
        set_wandb(dir_name, variant)


def get_snapshot_dir():
    return _snapshot_dir


def get_snapshot_mode():
    return _snapshot_mode


def set_snapshot_mode(mode):
    global _snapshot_mode
    _snapshot_mode = mode


def get_snapshot_gap():
    return _snapshot_gap


def set_snapshot_gap(gap):
    global _snapshot_gap
    _snapshot_gap = gap


def set_log_tabular_only(log_tabular_only):
    global _log_tabular_only
    _log_tabular_only = log_tabular_only


def get_log_tabular_only():
    return _log_tabular_only


def set_log_tboard(log_tboard):
    global _log_tboard
    _log_tboard = log_tboard


def get_log_tboard():
    return _log_tboard


def record_tboard(key, x, y, **kwargs) -> None:
    _summary_writer.add_scalar(key, y, global_step=x)


def record_wandb(key, x, y, **kwargs) -> None:
    wandb.log({key: y}, step=int(x))


def log(s, with_prefix=True, with_timestamp=True):
    out = s
    if with_prefix:
        out = _prefix_str + out
    if with_timestamp:
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f %Z")
        out = "%s | %s" % (timestamp, out)
    if not _log_tabular_only:
        # Also log to stdout
        print(out)
        for fd in list(_text_fds.values()):
            fd.write(out + "\n")
            fd.flush()
        sys.stdout.flush()


def record_tabular(key, val):
    _tabular.append((_tabular_prefix_str + str(key), str(val)))


def push_tabular_prefix(key):
    _tabular_prefixes.append(key)
    global _tabular_prefix_str
    _tabular_prefix_str = "".join(_tabular_prefixes)


def pop_tabular_prefix():
    del _tabular_prefixes[-1]
    global _tabular_prefix_str
    _tabular_prefix_str = "".join(_tabular_prefixes)


def save_extra_data(data, name="extra_data.pkl"):
    """
    Data saved here will always override the last entry

    :param data: Something pickle'able.
    """
    file_name = osp.join(_snapshot_dir, name)
    joblib.dump(data, file_name, compress=3)


def get_table_dict():
    return dict(_tabular)


def get_table_key_set():
    return set(key for key, value in _tabular)


@contextmanager
def prefix(key):
    push_prefix(key)
    try:
        yield
    finally:
        pop_prefix()


@contextmanager
def tabular_prefix(key):
    push_tabular_prefix(key)
    yield
    pop_tabular_prefix()


class TerminalTablePrinter(object):
    def __init__(self):
        self.headers = None
        self.tabulars = []

    def print_tabular(self, new_tabular):
        if self.headers is None:
            self.headers = [x[0] for x in new_tabular]
        else:
            assert len(self.headers) == len(new_tabular)
        self.tabulars.append([x[1] for x in new_tabular])
        self.refresh()

    def refresh(self):
        rows, columns = os.popen("stty size", "r").read().split()
        tabulars = self.tabulars[-(int(rows) - 3) :]
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(tabulate(tabulars, self.headers))
        sys.stdout.write("\n")


table_printer = TerminalTablePrinter()


def dump_tabular(*args, **kwargs):
    wh = kwargs.pop("write_header", None)
    if len(_tabular) > 0:
        if _log_tabular_only:
            table_printer.print_tabular(_tabular)
        else:
            for line in tabulate(_tabular).split("\n"):
                log(line, *args, **kwargs)
        tabular_dict = dict(_tabular)
        # Also write to the csv files
        # This assumes that the keys in each iteration won't change!
        for tabular_fd in list(_tabular_fds.values()):
            writer = csv.DictWriter(tabular_fd, fieldnames=list(tabular_dict.keys()))
            if wh or (wh is None and tabular_fd not in _tabular_header_written):
                writer.writeheader()
                _tabular_header_written.add(tabular_fd)
            writer.writerow(tabular_dict)
            tabular_fd.flush()

        del _tabular[:]

        # Try to write in tensorboard
        if _log_tboard:
            assert _summary_writer is not None, "summary writer is none!"
            step = tabular_dict[_step_key]
            for key in tabular_dict.keys():
                if key != _step_key:
                    record_tboard(key, step, np.array(tabular_dict[key]))

        if _log_wandb:
            assert wandb.run is not None, "wandb is not setup"
            step = tabular_dict[_step_key]
            for key in tabular_dict.keys():
                if key != _step_key:
                    record_wandb(
                        key, step, np.array(tabular_dict[key], dtype=np.float32)
                    )


def pop_prefix():
    del _prefixes[-1]
    global _prefix_str
    _prefix_str = "".join(_prefixes)


def save_itr_params(itr, params):
    if _snapshot_dir:
        if _snapshot_mode == "all":
            file_name = osp.join(_snapshot_dir, "itr_%d.pkl" % itr)
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "last":
            # override previous params
            file_name = osp.join(_snapshot_dir, "params.pkl")
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "gap":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, "itr_%d.pkl" % itr)
                joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "gap_and_last":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, "itr_%d.pkl" % itr)
                joblib.dump(params, file_name, compress=3)
            file_name = osp.join(_snapshot_dir, "params.pkl")
            joblib.dump(params, file_name, compress=3)
        elif _snapshot_mode == "none":
            pass
        else:
            raise NotImplementedError


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {"$class": o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {"$enum": o.__module__ + "." + o.__class__.__name__ + "." + o.name}
        return json.JSONEncoder.default(self, o)


def log_variant(log_file, variant_data):
    mkdir_p(os.path.dirname(log_file))
    with open(log_file, "w") as f:
        json.dump(variant_data, f, indent=2, sort_keys=True, cls=MyEncoder)


def record_tabular_misc_stat(key, values, placement="back"):
    if placement == "front":
        prefix = ""
        suffix = key
    else:
        prefix = key
        suffix = ""
    if len(values) > 0:
        record_tabular(prefix + "Average" + suffix, np.average(values))
        record_tabular(prefix + "Std" + suffix, np.std(values))
        record_tabular(prefix + "Median" + suffix, np.median(values))
        record_tabular(prefix + "Min" + suffix, np.min(values))
        record_tabular(prefix + "Max" + suffix, np.max(values))
    else:
        record_tabular(prefix + "Average" + suffix, np.nan)
        record_tabular(prefix + "Std" + suffix, np.nan)
        record_tabular(prefix + "Median" + suffix, np.nan)
        record_tabular(prefix + "Min" + suffix, np.nan)
        record_tabular(prefix + "Max" + suffix, np.nan)
