import os

# Used by wandb
PROJECT_NAME = "PADIL"

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

LOCAL_LOG_DIR = os.path.join(BASE_DIR, "logs")

OUTPUT_DIR_FOR_DOODAD_TARGET = os.path.join(BASE_DIR, "output")

NON_CODE_MOUNTS = None
