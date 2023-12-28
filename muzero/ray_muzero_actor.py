import logging
import ray
import reverb
import distutils.dir_util
import os
import jax
import adder
import atari_env_loop
import config
import networks
import specs
import utils
from actor import MzActor

from ray_inference import RayInferenceActor
from typing import Optional
import socket

jax.config.update("jax_platform_name", "cpu")


# TODO - merge MzActor with RayMuzeroActor
@ray.remote(num_cpus=1, resources={"actor": 1})
class RayMuzeroActor:
    """Ray actor wrapper for the Muzero actor.
    Attributes:
        reverb_server_address: The address of the Reverb server.
        environment_name: The name of the Atari environment
        save_dir: The directory to save logs.
        is_evaluator: Whether or not this actor is being used for evaluation.
        actor_id: The index of the actor.
        ckpt_dir: The directory where to save checkpoints.
        tensorboard_dir: The directory where to save tensorboard summaries.
        use_priority_fn: Whether or not to use the priority function.
        inference_actor: The optional TPU inference actor for which to send
            and receive inference requests to/from.
    """

    def __init__(
        self,
        reverb_server_address: str,
        environment_name: str,
        save_dir: str,
        is_evaluator: bool,
        actor_id: int,
        ckpt_dir: str,
        tensorboard_dir: str,
        use_priority_fn: bool,
        inference_actor_dyna: Optional[RayInferenceActor] = None,
        inference_actor_repr: Optional[RayInferenceActor] = None,
    ):
        logging.info("OS Environment: %s", os.environ)
        self.reverb_server_address = reverb_server_address
        self.environment_name = environment_name
        self.save_dir = save_dir
        self.is_evaluator = is_evaluator
        self.actor_id = actor_id
        self.ckpt_dir = ckpt_dir
        self.tensorboard_dir = tensorboard_dir
        self.use_priority_fn = use_priority_fn

        if inference_actor_dyna is not None:
            self.inference_actor_dyna = inference_actor_dyna
        else:
            self.inference_actor_dyna = None
        if inference_actor_repr is not None:
            self.inference_actor_repr = inference_actor_repr
        else:
            self.inference_actor_repr = None

    def live(self) -> bool:
        return True

    def initialize(self):
        model_config = config.ModelConfig()
        # optim_config = config.OptimConfig()
        train_config = config.TrainConfig()
        replay_config = config.ReplayConfig()

        key1, _ = jax.random.split(jax.random.PRNGKey(self.actor_id))

        environment = atari_env_loop.make_atari_environment(self.environment_name)
        environment_specs = specs.make_environment_spec(environment)
        input_specs = atari_env_loop.get_environment_spec_for_init_network(
            self.environment_name
        )
        extra_specs = atari_env_loop.get_extra_spec()

        model = networks.get_model(model_config)

        _unroll_step = max(train_config.td_steps, train_config.num_unroll_steps)
        reverb_client = reverb.Client(self.reverb_server_address)

        if self.use_priority_fn:
            priority_fn = utils.compute_td_priority
        else:
            priority_fn = None

        atari_adder = adder.SequenceAdder(
            client=reverb_client,
            # the period should be shorter than replay seq length in order to
            # include all possible starting points. Particularly the margin should
            # be at least stack_frame + unroll + 1 to include everything
            # to be safe we also * 2 here, but in the most rigorous way, we should
            # not * 2.
            # TODO(wendyshang): test not to * 2
            period=replay_config.replay_sequence_length
            - (atari_env_loop.ATARI_NUMBER_STACK_FRAME + _unroll_step + 1) * 2,
            sequence_length=replay_config.replay_sequence_length,
            end_of_episode_behavior=adder.EndBehavior.WRITE,
            environment_spec=environment_specs,
            extras_spec=extra_specs,
            init_padding=atari_env_loop.ATARI_NUMBER_STACK_FRAME - 1,
            end_padding=_unroll_step,
            priority_fns={adder.DEFAULT_PRIORITY_TABLE: priority_fn},
        )

        mz_actor = MzActor(
            network=model,
            actor_id=self.actor_id,
            observation_spec=input_specs,
            rng=key1,
            ckpt_dir=self.ckpt_dir,
            ckpt_save_interval_steps=train_config.ckpt_save_interval_steps,
            adder=atari_adder,
            mcts_params=None,  # TODO: change!
            use_argmax=False,
            use_mcts=True,
            inference_dyna_actor=self.inference_actor_dyna,
            inference_repr_actor=self.inference_actor_repr,
            total_training_steps=train_config.total_training_steps,
        )
        if self.actor_id == 0:
            reverb_client = self.reverb_server_address
        else:
            reverb_client = ""

        self.env_loop = atari_env_loop.AtariEnvironmentLoop(
            environment=environment,
            actor=mz_actor,
            reverb_client=reverb_client,
            tensorboard_dir=self.tensorboard_dir,
        )

    def run(self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None):
        logging.info("Running the environment loop.")
        self.env_loop.run(num_episodes=num_episodes, num_steps=num_steps)
        if self.save_dir != "":
            print("copying tmp log to persistent nsf")
            distutils.dir_util.copy_tree(
                "/tmp/ray/session_latest/logs", os.path.join(self.save_dir, "act")
            )

    def __repr__(self):
        """Prints out formatted Ray actor logs."""
        return f"[MuZeroActor: {self.actor_id}]({socket.gethostname()})"
