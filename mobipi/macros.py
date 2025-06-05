"""
Macros.
"""
import os

MOBIPI_DIR = os.path.dirname(os.path.dirname(__file__))

SCENE_MODEL_ROOT_DIR = os.path.join(MOBIPI_DIR, "scene_data")

POLICY_CKPT_ROOT_DIR = os.path.join(MOBIPI_DIR, "ckpts")

LOG_ROOT_DIR = os.path.join(MOBIPI_DIR, "logs")

DATA_ROOT_DIR = os.path.join(MOBIPI_DIR, "data")


try:
    from mobipi.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    import mobipi
    log_warning(
        "No private macro file found!"\
        "\nIt is recommended to use a private macro file"\
        "\nTo setup, run: python {}/scripts/setup_macros.py".format(mobipi.__path__[0])
    )
