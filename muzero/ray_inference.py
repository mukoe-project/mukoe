from typing import List, Any, Iterable
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax._src.mesh import Mesh
from jax.sharding import PositionalSharding

import logging
import socket

import ray

import networks
import config
import checkpoint_utils
import array_encode_decode

from batching import RequestBatcher


@ray.remote(resources={"inference_cpu_handler": 1})
class RayInferenceActor:
    def __init__(
        self,
        ckpt_dir: str,
        batch_size: int = 256,
        model="repr",
        batch_timeout_s: float = 1.0,
        tpu_id: str = "inference",
        weight_update_interval: int = 100,
    ):
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_s
        self.ckpt_dir = ckpt_dir
        self.tpu_id = tpu_id
        self.step_counter = 0
        self.weight_update_interval = weight_update_interval
        self._model = model
        if model == "repr":
            self.actor_def = RayReprInferenceShard
        elif model == "dyna":
            self.actor_def = RayDynaInferenceShard
        else:
            self._actor_def = None
            print("Invalid model provided...")

    def live(self) -> bool:
        if not self._shards:
            # Init has probably not started yet
            return True
        try:
            shards_live = ray.get([shard.live.remote()] for shard in self._shards)
            return all(shards_live)
        except Exception:
            return False

    def initialize(self):
        if not self.actor_def:
            raise ValueError("Actor def is not defined. Check the provided model.")
        if "v4_16" in self.tpu_id:
            num_hosts = 2
        else:
            num_hosts = 1

        logging.info("Number of hosts: %d", num_hosts)
        print("Number of hosts: ", num_hosts)
        self._shards = [
            self.actor_def.options(
                max_concurrency=2, resources={self.tpu_id: 1, "TPU": 4}
            ).remote(ckpt_dir=self.ckpt_dir)
            for _ in range(num_hosts)
        ]
        init_handles = [shard.initialize.remote() for shard in self._shards]
        self._batcher = RequestBatcher.remote(
            batch_handler_fn=self.process_batch,
            batch_size=self.batch_size,
            batch_timeout_s=self.batch_timeout_s,
        )
        ray.get(init_handles)

    def process_batch(self, inputs: Iterable[Any]):
        print("Processing batch")
        try:
            if (
                self.step_counter > 0
                and self.step_counter % self.weight_update_interval == 0
            ):
                print("updating weights")
                start_time = time.time()
                ray.get([shard.update_weights.remote() for shard in self._shards])
                print(f"update weight time time {time.time() - start_time}s")
            results = ray.get(
                [shard.handle_batch.remote(inputs) for shard in self._shards]
            )
            final_result = results[0]
            self.step_counter += 1
            # Results will be a list of list of results
            # We will need to flatten this when we have more than one host
            # but for now let's just return the first index in single host
            # case.
            # check this again for multihost
            return final_result

        except Exception as e:
            print("process_batch failed due to: ", e)
            raise e

    def put(self, input: Any) -> str:
        return ray.get(self._batcher.put.remote(input))

    def get(self, request_id: str) -> Any:
        return ray.get(self._batcher.get.remote(request_id))


class RayInferenceShardBase:
    """Base class for Ray Inference Shards."""

    def __init__(self, ckpt_dir: str, skip_checkpoint: bool = False):
        """Ray-based Dynamic inferencer."""
        self.ckpt_dir = ckpt_dir
        self._skip_checkpoint = skip_checkpoint

    def live(self) -> bool:
        return True

    def update_weights(self) -> None:
        print("beginning weight update")
        all_steps = self._ckpt_manager.all_steps(read=True)
        latest_step = max(all_steps) if all_steps else None
        logging.info(f"actor latest_ckpt_step={latest_step}")
        print(f"actor latest_ckpt_step={latest_step}")
        self.step = latest_step
        if latest_step:
            count_try = 0
            while True:
                if count_try > 3:
                    return False
                try:
                    print("Trying to restore checkpoint")
                    restored = self._ckpt_manager.restore(latest_step)
                    print("Initial restore complete")
                    restored_params = restored["save_state"]["state"]
                    restored_step = restored_params["step"]
                    print("Restored step ", restored_step)
                    logging.info(f"actor restored_ckpt_step={restored_step}")
                    self._latest_step = restored_step
                    self._model_params_states = restored_params
                    if self._latest_step - 1 >= self._total_training_steps:
                        self._finished = True
                    return True
                except Exception:
                    count_try += 1
                    logging.info("waiting for 30s and retry updating actor.")
                    time.sleep(30)
        else:
            return False


@ray.remote
class RayDynaInferenceShard(RayInferenceShardBase):
    def __repr__(self):
        return f"[RayDynaInferenceActorShard:{socket.gethostname()}]"

    def initialize(self):
        model_config = config.ModelConfig()
        network = networks.get_model(model_config)

        def dyna_and_pred(params, embedding, action):
            return network.apply(
                params, embedding, action, method=network.dyna_and_pred
            )

        self._jitted_dyna_and_pred = jax.jit(dyna_and_pred)
        self._num_devices = jax.device_count()
        self._emb_sharding = PositionalSharding(jax.devices()).reshape(
            self._num_devices, 1, 1, 1
        )
        self._action_sharding = PositionalSharding(jax.devices()).reshape(
            self._num_devices
        )
        self._mesh = Mesh(np.asarray(jax.devices(), dtype=object), ["data"])

        if self._skip_checkpoint:
            print("Skipping checkpoint loading...")
            _, key2 = jax.random.split(jax.random.PRNGKey(42))
            dummy_obs = jnp.zeros((1, 96, 96, 3, 4), dtype=jnp.float32)
            dummy_action = jnp.zeros((1, 1), dtype=jnp.int32)
            params = network.init(key2, dummy_obs, dummy_action)
            self._model_params_states = params
        else:
            dummy_ckpt_save_interval_steps = 10

        while True:
            try:
                self._ckpt_manager = checkpoint_utils.get_ckpt_manager(
                    self.ckpt_dir,
                    dummy_ckpt_save_interval_steps,
                    create=False,
                    use_async=False,
                )
                print("got ckpt manager")
                break
            except Exception:
                print("waiting for 30s and retry.")
                time.sleep(30)
        all_steps = self._ckpt_manager.all_steps(read=True)
        latest_step = max(all_steps) if all_steps else None
        if latest_step is None:
            latest_step = 0
            print(f"need to load actor latest_ckpt_step={latest_step}")
        while True:
            try:
                restored = self._ckpt_manager.restore(latest_step)
                restored_params = restored["save_state"]["state"]
                self._model_params_states = restored_params
                print("done restoring")
                break
            except Exception:
                print(f"trying to load {latest_step} again")
                time.sleep(30)

    def batch_and_shard(self, inputs: Iterable[Any]):
        embeddings = []
        actions = []
        for embedding, action in inputs:
            embedding = array_encode_decode.ndarray_from_bytes(embedding)
            action = array_encode_decode.ndarray_from_bytes(action)
            embeddings.append(embedding)
            actions.append(action)

        num_to_pad = (
            self._num_devices - (len(inputs) % self._num_devices)
        ) % self._num_devices
        for i in range(num_to_pad):
            embeddings.append(array_encode_decode.ndarray_from_bytes(inputs[0][0]))
            actions.append(array_encode_decode.ndarray_from_bytes(inputs[0][1]))

        global_embedding = np.concatenate(embeddings, axis=0)
        global_embedding_shape = global_embedding.shape
        embedding_arrays = [
            jax.device_put(global_embedding[index], d)
            for d, index in self._emb_sharding.addressable_devices_indices_map(
                global_embedding_shape
            ).items()
        ]
        sharded_embedding = jax.make_array_from_single_device_arrays(
            global_embedding_shape, self._emb_sharding, embedding_arrays
        )

        global_actions = np.concatenate(actions, axis=0)
        global_actions_shape = global_actions.shape
        action_arrays = [
            jax.device_put(global_actions[index], d)
            for d, index in self._action_sharding.addressable_devices_indices_map(
                global_actions_shape
            ).items()
        ]
        sharded_actions = jax.make_array_from_single_device_arrays(
            global_actions_shape, self._action_sharding, action_arrays
        )
        return (sharded_embedding, sharded_actions)

    def handle_batch(self, inputs: Iterable[Any]) -> List[Any]:
        def print_shape_and_type(x):
            return x.shape, x.dtype

        embedding, action = self.batch_and_shard(inputs)
        dp_net_out = self._jitted_dyna_and_pred(
            {"params": self._model_params_states["params"]}, embedding, action
        )
        jax.block_until_ready(dp_net_out)
        dp_net_out = jax.experimental.multihost_utils.process_allgather(
            dp_net_out, tiled=True
        )
        result = dp_net_out[1]
        result["embedding"] = dp_net_out[0]
        result_list = jax.tree_util.tree_map(list, result)
        final_result = []
        for i in range(len(inputs)):
            result = {
                key: array_encode_decode.ndarray_to_bytes(
                    np.asarray(result_list[key][i])
                )
                for key in result_list
            }
            final_result.append(result)
        return final_result


@ray.remote
class RayReprInferenceShard(RayInferenceShardBase):
    def __repr__(self):
        return f"[RayReprInferenceActorShard:{socket.gethostname()}]"

    def initialize(self):
        model_config = config.ModelConfig()
        network = networks.get_model(model_config)

        def repr_and_pred(params, obs, dtype=jnp.float32):
            return network.apply(params, obs, dtype, method=network.repr_and_pred)

        self._jitted_repr_and_pred = jax.jit(repr_and_pred)
        self._num_devices = jax.device_count()
        self._obs_sharding = PositionalSharding(jax.devices()).reshape(
            self._num_devices, 1, 1, 1, 1
        )
        self._mesh = Mesh(np.asarray(jax.devices(), dtype=object), ["data"])
        dummy_ckpt_save_interval_steps = 10
        while True:
            try:
                print("try to get ckpt manager")
                self._ckpt_manager = checkpoint_utils.get_ckpt_manager(
                    self.ckpt_dir,
                    dummy_ckpt_save_interval_steps,
                    create=False,
                    use_async=False,
                )
                print("got ckpt manager")
                break
            except Exception:
                print("waiting for 30s and reetry.")
                time.sleep(30)
        all_steps = self._ckpt_manager.all_steps(read=True)
        latest_step = max(all_steps) if all_steps else None
        self.step = latest_step
        if latest_step is None:
            latest_step = 0
            print(f"need to load actor latest_ckpt_step={latest_step}")
        while True:
            try:
                restored = self._ckpt_manager.restore(latest_step)
                restored_params = restored["save_state"]["state"]
                self._model_params_states = restored_params
                print("done restoring")
                break
            except Exception:
                print(f"trying to load {latest_step} again")
                time.sleep(30)

    def batch_and_shard(self, inputs: Iterable[any]):
        observations = []
        for observation in inputs:
            observation = array_encode_decode.ndarray_from_bytes(observation)
            observations.append(observation)
        num_to_pad = (
            self._num_devices - (len(inputs) % self._num_devices)
        ) % self._num_devices
        for i in range(num_to_pad):
            observations.append(array_encode_decode.ndarray_from_bytes(inputs[0]))
        global_observation = np.concatenate(observations, axis=0)
        global_observation_shape = global_observation.shape
        observation_arrays = [
            jax.device_put(global_observation[index], d)
            for d, index in self._obs_sharding.addressable_devices_indices_map(
                global_observation_shape
            ).items()
        ]
        shared_observation = jax.make_array_from_single_device_arrays(
            global_observation_shape, self._obs_sharding, observation_arrays
        )
        return shared_observation

    def handle_batch(self, inputs: Any) -> List[Any]:
        def print_shape_and_type(x):
            return x.shape, x.dtype

        observation = self.batch_and_shard(inputs)
        repr_net_out = self._jitted_repr_and_pred(
            {"params": self._model_params_states["params"]}, observation
        )
        jax.block_until_ready(repr_net_out)
        repr_net_out = jax.experimental.multihost_utils.process_allgather(
            repr_net_out, tiled=True
        )
        result = repr_net_out[1]
        result["embedding"] = repr_net_out[0]
        result_list = jax.tree_util.tree_map(list, result)
        final_result = []
        for i in range(len(inputs)):
            result = {
                key: array_encode_decode.ndarray_to_bytes(result_list[key][i])
                for key in result_list
            }
            result["step"] = self.step
            final_result.append(result)
        return final_result
