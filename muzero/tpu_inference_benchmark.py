"""Simple script that we can use for benchmarking RayTpuInference."""
import logging
import ray
import argparse
import traceback
import getpass
import jax
import jax.numpy as jnp
import numpy as np
import time
import event_logger
import array_encode_decode

from ray_inference import RayInferenceActor

TPU_NAME = "trainer"
TPU_ID = "inference_v4_8"


def setup_loggers():
    logger = logging.getLogger("ray")
    logger.setLevel(logging.INFO)


def main_dyna(args: argparse.ArgumentParser):
    logging.info("Starting inference benchmarker...")
    logging.info("Args: %s", args)
    ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
    event_logger.initialize()

    batch_size = int(args.batch_size)
    batch_timeout_s = float(args.batch_timeout_s)
    num_requests = int(args.num_requests)
    weight_update_interval = int(args.weight_update_interval)

    try:
        start_time = time.time()
        inference_actor = RayInferenceActor.options(resources={TPU_NAME: 1}).remote(
            model="dyna",
            ckpt_dir=args.ckpt_dir,
            batch_size=batch_size,
            batch_timeout_s=batch_timeout_s,
            tpu_id=TPU_ID,
            weight_update_interval=weight_update_interval)
        logging.info("Initializing inference actor.")
        ray.get(inference_actor.initialize.remote())
        logging.info("Took %f seconds to initialize RayTpuInferenceActor...", time.time() - start_time)
        logging.info("Starting to run %d requests: ", num_requests) 

        embedding = np.random.normal(0, 1, size=(1, 6, 6, 256))
        action = np.zeros((1,))
        logging.info("Sending requests...")
        # convert embedding and action to bytes 
        embedding = array_encode_decode.ndarray_to_bytes(embedding)
        action = array_encode_decode.ndarray_to_bytes(action)

        start_time = time.time()
        handles = [
            ray.get(inference_actor.put.remote((embedding, action))) for _ in range(num_requests)
        ]
        logging.info("Sent requests in %f seconds", time.time() - start_time)
        start_time = time.time()
        logging.info("Getting results...")
        logging.info("Got requests in %f seconds", time.time() - start_time)
        results_handle = [inference_actor.get.remote(handle) for handle in handles]
        final_results = [ray.get(result_handle) for result_handle in results_handle]
        final_result = final_results[0]
        end_time = time.time()
        for k in final_result:
            final_result[k] = array_encode_decode.ndarray_from_bytes(final_result[k])
        print(final_result)

        logging.info("Took %f seconds to run %d iterations for an average of %f seconds / it.",
                    end_time - start_time, num_requests, (end_time - start_time) / num_requests)
        # best previous was 0.006418328762054443 sec per request
    except Exception as e:
        logging.info("Caught error while running: %s. Shutting down",
                     traceback.format_exc())
        ray.shutdown()
        exit(1)

    print("Printing summary...")
    event_logger.summary()
    logging.info("All done!")
    time.sleep(5)
    event_logger.teardown()
    ray.shutdown()


def main_repr(args: argparse.ArgumentParser):
    logging.info("Starting inference benchmarker...")
    logging.info("Args: %s", args)
    ray.init(runtime_env=dict(worker_process_setup_hook=setup_loggers))
    event_logger.initialize()

    batch_size = int(args.batch_size)
    batch_timeout_s = float(args.batch_timeout_s)
    num_requests = int(args.num_requests)
    weight_update_interval = int(args.weight_update_interval)

    try:
        start_time = time.time()
        inference_actor = RayInferenceActor.options(resources={TPU_NAME: 1}).remote(
            ckpt_dir=args.ckpt_dir,
            model="repr",
            batch_size=batch_size,
            batch_timeout_s=batch_timeout_s,
            tpu_id=TPU_ID,
            weight_update_interval=weight_update_interval)
        logging.info("Initializing inference actor.")
        ray.get(inference_actor.initialize.remote())
        logging.info("Took %f seconds to initialize RayTpuInferenceActor...", time.time() - start_time)
        logging.info("Starting to run %d requests: ", num_requests) 

        observation = np.random.normal(0, 1, size=(1, 96, 96, 3, 4))
        logging.info("Sending requests...")
        # convert embedding and action to bytes 
        observation = array_encode_decode.ndarray_to_bytes(observation)

        start_time = time.time()
        handles = [
            ray.get(inference_actor.put.remote(observation)) for _ in range(num_requests)
        ]
        logging.info("Sent requests in %f seconds", time.time() - start_time)
        logging.info("Getting results...")
        results_handle = [inference_actor.get.remote(handle) for handle in handles]
        final_results = [ray.get(result_handle) for result_handle in results_handle]
        final_result = final_results[0]
        end_time = time.time()
        for k in final_result:
            if k != "step":
                final_result[k] = array_encode_decode.ndarray_from_bytes(final_result[k])
        print(final_result)

        logging.info("Took %f seconds to run %d iterations for an average of %f seconds / it.",
                    end_time - start_time, num_requests, (end_time - start_time) / num_requests)
        # best previous was 0.006418328762054443 sec per request
    except Exception as e:
        logging.info("Caught error while running: %s. Shutting down",
                     traceback.format_exc())
        ray.shutdown()
        exit(1)

    print("Printing summary...")
    event_logger.summary()
    logging.info("All done!")
    time.sleep(5)
    event_logger.teardown()
    ray.shutdown()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        prog="Ray inference benchmarker",
        description="Inference actor benchmarker")
    parser.add_argument("--num_requests", action="store", default=1000)
    parser.add_argument("--ckpt_dir", action="store", default="")
    parser.add_argument("--batch_size", action="store", default=256)
    parser.add_argument("--batch_timeout_s", action="store", default=1.0)
    parser.add_argument("--inference_type", action="store", default="")
    parser.add_argument("--weight_update_interval", action="store", default=1000)
    
    args = parser.parse_args()
    if args.inference_type == "dyna":
        main_dyna(args=args)
    elif args.inference_type == "repr":
        main_repr(args=args)
    else:
        assert "Wrong Inference Type"
