import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_manager import (
    CheckpointManagerOptions,
    Checkpointer,
    AsyncCheckpointer,
)
import socket
import jax
from jax.experimental import multihost_utils
import portpicker
import logging


def _multislice_distribute_initialize():
    """Calls jax.distribute.initialize() with appropriate multislice arguments."""

    def gen_local_ip():
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

    def gen_local_ip_nums():
        return [int(num) for num in gen_local_ip().split(":")[-1].split(".")]

    def get_coordinator_ip():
        local_ip_nums = jax.numpy.array(gen_local_ip_nums())
        coordinator_ip_nums = multihost_utils.broadcast_one_to_all(local_ip_nums)
        coordinator_ip_strings = [str(num) for num in list(coordinator_ip_nums)]
        return ".".join(coordinator_ip_strings)

    port = multihost_utils.broadcast_one_to_all(
        jax.numpy.array(portpicker.pick_unused_port())
    )
    coordinator_address = get_coordinator_ip() + ":" + str(port)
    try:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=jax.process_count(),
            process_id=jax.process_index(),
        )
    except RuntimeError:
        logging.info("Jax distributed already initialized")
        pass


def get_ckpt_manager(path, save_interval_steps, create=True, use_async=True):
    # p = epath.Path(path)
    options = CheckpointManagerOptions(
        create=create, max_to_keep=5, save_interval_steps=save_interval_steps
    )
    checkpointer = ocp.PyTreeCheckpointHandler()
    if use_async:
        _multislice_distribute_initialize()
        checkpointer = AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
    else:
        checkpointer = Checkpointer(ocp.PyTreeCheckpointHandler())
    mngr = ocp.CheckpointManager(
        path,
        {
            "save_state": checkpointer,
        },
        options=options,
    )
    logging.info("ckpt manager created!")
    return mngr
