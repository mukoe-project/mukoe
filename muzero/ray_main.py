import logging
import ray
import argparse
import getpass
import traceback
import reverb
from ray_reverb import RayReverbServer
import config

import event_logger
from ray_muzero_actor import RayMuzeroActor
from ray_train import RayTpuTrainer
from ray_inference import RayInferenceActor

TPU_ID_REPR = "inference_v4_8_repr"
TPU_ID_DYNA = "inference_v4_8_dyna"
_CPU_RESOURCE_STR = "inference_cpu_handler"
_EVENT_LOGGER_FREQUENCY = 5


def setup_loggers():
    logging.basicConfig(level=logging.INFO)


class MuzeroRunner:
    def __init__(self, args: argparse.ArgumentParser):
        # TODO - expand out the args
        self.args = args
        self.actors = None
        self.actor_index_map = None
        self.trainer_actor = None
        self.reverb_actor = None
        self.inference_actor_dyna = None
        self.inference_actor_repr = None
        self.reverb_future = None

    def start_reverb(self):
        args = self.args
        logging.info("Initializing reverb server...")
        self.reverb_actor = RayReverbServer.remote(
            environment_name=args.environment, port=9090, reverb_dir=args.reverb_dir
        )
        # Initialize reverb actor and wait until ready.
        ray.get(self.reverb_actor.initialize.remote())
        self.reverb_server_address = ray.get(self.reverb_actor.get_ip.remote())
        logging.info("Reverb server address is: %s", self.reverb_server_address)
        self.reverb_client = reverb.Client(self.reverb_server_address)
        logging.info(self.reverb_client.server_info())
        print(self.reverb_client.server_info())
        self.reverb_future = self.reverb_actor.start.remote()

    def start_learner(self):
        args = self.args
        logging.info("Initializing trainer...")
        self.trainer_actor = RayTpuTrainer.options(
            resources={_CPU_RESOURCE_STR: 0.1}
        ).remote(
            reverb_server_address=self.reverb_server_address,
            save_dir=args.save_dir,
            ckpt_dir=args.ckpt_dir,
            tensorboard_dir=args.tensorboard_dir,
            environment_name=args.environment,
        )
        trainer_init_handle = self.trainer_actor.initialize.remote()
        # learner has to be first initialized
        ray.get(trainer_init_handle)

    def start_inference(self):
        args = self.args
        logging.info("Initializing inference actors...")
        if args.inference_node == "tpu":
            assert "tpu inference is incomplete"
            inference_config = config.InferenceConfig
            # TODO(wendyshang): add inference_actor_dyna and inference_actor_repr
            self.inference_actor_dyna = RayInferenceActor.options(
                resources={_CPU_RESOURCE_STR: 0.1}
            ).remote(
                ckpt_dir=args.ckpt_dir,
                batch_size=inference_config.dyna_batch_size,
                batch_timeout_s=inference_config.dyna_time_out,
            )
            self.inference_actor_repr = RayInferenceActor.options(
                resources={_CPU_RESOURCE_STR: 0.1}
            ).remote(
                ckpt_dir=args.ckpt_dir,
                batch_size=inference_config.repr_batch_size,
                batch_timeout_s=inference_config.repr_time_out,
            )
            logging.info("Initializing inferencer")
            inference_dyna_init_handle = self.inference_actor_dyna.initialize.remote()
            inference_repr_init_handle = self.inference_actor_repr.initialize.remote()
            ray.get([inference_dyna_init_handle, inference_repr_init_handle])
        elif args.inference_node == "cpu":
            self.inference_actor_dyna = None
            self.inference_actor_repr = None
        else:
            assert "wrong inference node type specified: cpu or tpu only"

    def start_muzero_actor(self, actor_i: int):
        args = self.args
        if actor_i % 100 == 0:
            action_i_save_dir = (
                ""  # this is the dir to save log, TODO(wendyshang): add dir
            )
            action_i_tb_dir = args.tensorboard_dir
        else:
            action_i_save_dir = ""
            action_i_tb_dir = ""
        runtime_env_actor = {
            "env_vars": {
                "JAX_BACKEND": "CPU",
                "JAX_PLATFORMS": "cpu",
                "GCS_RESOLVE_REFRESH_SECS": "60",
                "RAY_memory_monitor_refresh_ms": "0",
            }
        }
        return RayMuzeroActor.options(
            num_cpus=args.core_per_task,
            runtime_env=runtime_env_actor,
            resources={"actor": 1},
        ).remote(
            reverb_server_address=self.reverb_server_address,
            environment_name=args.environment,
            save_dir=action_i_save_dir,
            is_evaluator=False,
            inference_actor_dyna=self.inference_actor_dyna,
            inference_actor_repr=self.inference_actor_repr,
            actor_id=actor_i,
            ckpt_dir=args.ckpt_dir,
            tensorboard_dir=action_i_tb_dir,
            use_priority_fn=False,
        )

    def start_muzero_actors(self):
        args = self.args
        logging.info(
            "Instantiating %d actors using %d CPUs per actor...",
            args.num_actors,
            args.core_per_task,
        )
        self.actors = []
        self.actor_index_map = {}
        for actor_i in range(args.num_actors):
            current_actor = self.start_muzero_actor(actor_i)
            self.actors.append(current_actor)
            self.actor_index_map[current_actor] = actor_i

        logging.info("Initializing MuZero Actors...")
        actor_init_handles = [a.initialize.remote() for a in self.actors]
        logging.info("Waiting for initialization to finish...")
        ray.get(actor_init_handles)

    def initialize(self):
        if "replay_buffer" not in ray.available_resources():
            raise ValueError(
                "While initializing, could not find a replay_buffer resource."
            )
        if "trainer" not in ray.available_resources():
            raise ValueError("While initializing, could not find a trainer resource.")
        try:
            event_logger.initialize()
            self.start_reverb()
            self.start_learner()
            self.start_inference()
            self.start_muzero_actors()
            logging.info("All actors are initialized and ready to run.")
        except Exception:
            logging.info(
                "Caught error during actor init: %s. Shutting down",
                traceback.format_exc(),
            )
            ray.shutdown()
            exit(1)

    def _start_runs(self):
        futures_to_actors = {}
        future = self.trainer_actor.train.remote()
        futures_to_actors[future] = self.trainer_actor
        futures_to_actors[self.reverb_future] = self.reverb_actor

        for actor in self.actors:
            future = actor.run.remote()
            futures_to_actors[future] = actor

        futures = list(futures_to_actors.keys())
        return futures, futures_to_actors

    def run_until_completion(self):
        results = []
        futures, futures_to_actors = self._start_runs()
        event_logger_counter = 0
        while futures:
            event_logger_counter += 1
            done_futures, futures = ray.wait(futures, timeout=5)
            if event_logger_counter % _EVENT_LOGGER_FREQUENCY == 0:
                event_logger.summary()
            if not done_futures:
                continue
            for future in done_futures:
                try:
                    print("Getting failed futures")
                    results.append(ray.get(future))
                except ray.exceptions.RayActorError:
                    failed_actor = futures_to_actors[future]
                    print(f"Actor {failed_actor} failed.")
                    if (
                        failed_actor == self.trainer_actor
                        or failed_actor == self.reverb_actor
                    ):
                        print("Detected that the trainer or reverb actor failed.")
                        print("Restarting the entire run...")
                        # If the trainer or the reverb actor fails, we need to restart
                        # everything
                        # TODO - consider what happens if some actors partially
                        # completed if applicable?
                        for actor in futures_to_actors.values():
                            ray.kill(actor)
                        self.initialize()
                        futures, futures_to_actors = self._start_runs()
                    else:
                        # TODO - once we incorporate TPU inference actors,
                        # add that here too.
                        print("Detected that a regular actor failed.")
                        print("Re-initializing and continuing the run...")
                        actor_i = self.actor_index_map[failed_actor]
                        self.actors.remove(failed_actor)
                        new_actor = self.start_muzero_actor(actor_i)
                        print("Initializing new actor...")
                        ray.get(new_actor.initialize.remote())
                        print("All done. Continuing the run.")
                        self.actors.append(new_actor)
                        self.actor_index_map[new_actor] = actor_i
                        futures.append(new_actor.run.remote())
                        futures_to_actors[future] = new_actor


def main(args: argparse.ArgumentParser):
    logging.info("Initializing the workload!")
    logging.info("Input args: %s", args)
    logging.info("Connecting to the Ray cluster.")

    ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))

    logging.info("Available Ray resources: %s", ray.available_resources())
    logging.info("All Ray resources: %s", ray.cluster_resources())

    runner = MuzeroRunner(args)
    runner.initialize()
    logging.info("Starting the workload...")
    runner.run_until_completion()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="Ray-RL-Demo",
        description="Our amazing MuZero implementation on Ray x TPUs!",
    )
    parser.add_argument("--ckpt_dir", action="store", default="")
    parser.add_argument(
        "--save_dir", action="store", default=f"/home/{getpass.getuser()}/ray-train"
    )
    parser.add_argument("--tensorboard_dir", action="store", default="")
    parser.add_argument("--reverb_dir", action="store", default="")
    parser.add_argument("--num_actors", action="store", type=int, default=10)
    parser.add_argument("--core_per_task", action="store", type=int, default=32)
    parser.add_argument("--environment", action="store", default="MsPacman")
    parser.add_argument("--inference_node", action="store", default="cpu")

    args = parser.parse_args()
    main(args=args)
