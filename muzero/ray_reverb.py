import logging
import ray

from typing import Optional

import socket

import specs
import config
import reverb
import atari_env_loop
import adder


def make_replay_table(
    min_size_to_sample: int,
    samples_per_insert: int,
    max_replay_size: int,
    name: str = adder.DEFAULT_PRIORITY_TABLE,
    signature: Optional[reverb.reverb_types.SpecNest] = None,
) -> reverb.Table:
    """The replay buffer for training."""
    # To make sure num_samples / num_inserts < samples_per_insert ratio,
    # we have:
    # 0 < num_inserts * samples_per_insert - num_samples
    # < max_replay_size * samples_per_insert
    samples_per_insert_tolerance = 0.1 * samples_per_insert
    error_buffer = min_size_to_sample * samples_per_insert_tolerance
    limiter = reverb.rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_size_to_sample,
        samples_per_insert=samples_per_insert,
        error_buffer=error_buffer,
    )
    # error_buffer=(0, max_replay_size * samples_per_insert))

    table = reverb.Table(
        name=name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=limiter,
        signature=signature,
    )
    return table


@ray.remote(resources={"replay_buffer": 1})
class RayReverbServer:
    def __init__(self, environment_name: str, port: int, reverb_dir: str):
        self.port = port
        self.environment_name = environment_name
        self.reverb_dir = reverb_dir

    def initialize(self):
        replay_config = config.ReplayConfig()
        environment = atari_env_loop.make_atari_environment(self.environment_name)
        environment_specs = specs.make_environment_spec(environment)
        extras_spec = atari_env_loop.get_extra_spec()
        signature = adder.SequenceAdder.signature(
            environment_specs,
            sequence_length=replay_config.replay_sequence_length,
            extras_spec=extras_spec,
        )
        max_replay_size = replay_config.max_replay_size
        samples_per_insert = replay_config.samples_per_insert
        min_size_to_sample = int(replay_config.min_fill_fraction * max_replay_size)
        replay_table = make_replay_table(
            min_size_to_sample, samples_per_insert, max_replay_size, signature=signature
        )
        if self.reverb_dir != "":
            logging.info(f"Setting reverb dir to be {self.reverb_dir}")
            checkpointer = reverb.checkpointers.DefaultCheckpointer(self.reverb_dir)
        else:
            checkpointer = None
        self.server = reverb.Server(
            tables=[replay_table], port=self.port, checkpointer=checkpointer
        )
        logging.info("Server info: %s", self.server.localhost_client().server_info())

    def start(self):
        self.server.wait()

    def server_info(self):
        return self.server.localhost_client().server_info()

    def get_ip(self):
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return f"{ip_address}:{self.port}"

    def __repr__(self):
        return "[ReverbServer]"
