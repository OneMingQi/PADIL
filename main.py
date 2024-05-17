# Import standard libraries
import os
import argparse
import sys
from time import sleep
from subprocess import Popen
import datetime
from pathlib import Path

# Import third-party libraries
import yaml
import dateutil

# Import project-specific modules
from algorithms.rlkit.launchers import config
from algorithms.rlkit.launchers.launcher_util import build_nested_variant_generator

# Uncomment the following line to specify the GPU to use, replacing '3' with the ID of the desired GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--experiment",
        default="./scripts/diffusion_disc.yaml",
        help="experimental parameter file",
    )
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    # args.nosrun = True

    # Load experiment specifications from YAML file
    with open(args.experiment, "r") as spec_file:
        exp_specs = yaml.safe_load(spec_file)

    # Generate the variants
    vg_fn = build_nested_variant_generator(exp_specs)

    # Create a directory to store all variants
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    variants_dir = (
        Path(config.LOCAL_LOG_DIR)
        / f"variants-for-{exp_specs['meta_data']['exp_name']}"
        / f"variants-{timestamp}"
    )
    variants_dir.mkdir(parents=True)

    # Write all variants to a file
    with open(os.path.join(variants_dir, "exp_spec_definition.yaml"), "w") as f:
        yaml.dump(exp_specs, f, default_flow_style=False)

    num_variants = 0
    for variant in vg_fn():
        variant["exp_id"] = num_variants
        with open((variants_dir / f"{num_variants}.yaml").as_posix(), "w") as f:
            yaml.dump(variant, f, default_flow_style=False)
        num_variants += 1

    # Determine the number of workers
    num_workers = min(exp_specs["meta_data"]["num_workers"], num_variants)
    exp_specs["meta_data"]["num_workers"] = num_workers

    # Prepare to run the processes
    running_processes = []
    args_idx = 0
    command = f"{sys.executable} {{script_path}} -e {{specs}} -g {{gpuid}}".split()
    command_format_dict = exp_specs["meta_data"]

    # Run the processes
    while (args_idx < num_variants) or (len(running_processes) > 0):
        if (len(running_processes) < num_workers) and (args_idx < num_variants):
            command_format_dict["specs"] = (
                variants_dir / f"{args_idx}.yaml"
            ).as_posix()
            command_format_dict["gpuid"] = args.gpu
            command_to_run = [part.format(**command_format_dict) for part in command]
            print(command_to_run)
            p = Popen(command_to_run)
            args_idx += 1
            running_processes.append(p)
        else:
            sleep(1)

        # Check the status of the running processes
        running_processes = [p for p in running_processes if p.poll() is None]
