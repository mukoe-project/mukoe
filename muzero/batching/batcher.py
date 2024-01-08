"""Utilities for synchronous batching mechanism.
This module provides a mechanism for aggregating separate, individual requests
(possibly from different Ray tasks or actors) into a single batch for optimal processing.
Once the batch processing is complete, results are disaggregated and returned to the respective requesters.
The functionality is conceptually similar to TensorFlow Serving's automatic batching,
Launchpad, or Ray Serve's @serve.batch feature.
See `batcher_test.py` for a reference of its usage.
"""
import ray
import time
import logging
import uuid
from ray.util.queue import Queue
import asyncio
from typing import Any, Callable, Iterable, List, Mapping, Optional


@ray.remote
def _batcher_task(
    request_queue: Queue, batch_queue: Queue, target_batch_size: int, timeout_in_s: int
):
    """Aggregates incoming requests continuously into batches.
    This task listens to a shared Ray queue (request_queue) for incoming requests,
    batches them based on the target batch size or a timeout, and then
    puts it into the batch queue.
    Args:
        request_queue: A Ray Queue that serves as the source of individual
            requests. Each item should be a tuple of (request_data, request_id).
        batch_queue: A Ray Queue which stores full batches. Each item should be
            a tuple of (request_data_batch, request_id-batch).
        target_batch_size: The ideal number of requests per batch. The task will
            try to fill each batch up to this size before invoking the batch_processor.
        timeout_in_s: Maximum time (in seconds) to wait for filling up a batch before
            invoking the batch_processor with whatever has been collected.
    Returns:
        None: This function runs indefinitely and returns nothing.
    """
    logging.debug("Starting batcher process...")
    print("Starting batcher process")

    while True:
        start_time = time.time()
        batch = []
        id_batch = []

        while (
            len(batch) < target_batch_size and time.time() - start_time < timeout_in_s
        ):
            try:
                timeout = min(timeout_in_s, time.time() - start_time)
                request_data, request_id = request_queue.get(timeout=timeout)
                logging.debug("Got request: %s with ID: %s", request_data, request_id)
                batch.append(request_data)
                id_batch.append(request_id)
                if len(batch) == target_batch_size:
                    logging.debug(
                        "Reached target batch size. Breaking the collection loop."
                    )
                    break
            except (asyncio.exceptions.TimeoutError, ray.util.queue.Empty) as e:
                logging.debug("Batcher task caught expected exception: %s", e)
                if len(batch) == 0:
                    break
                else:
                    continue
        if batch:
            result = (batch, id_batch)
            logging.debug("Putting in tuple %s into the batch queue", result)
            batch_queue.put(result)


@ray.remote
def _batch_processor_task(
    batch_queue: Queue,
    result_map: Mapping[Any, Any],
    batch_processor: Callable[[Iterable[Any]], List[Any]],
):
    """Runs processing on generated batches.

    This task listens to a shared Ray queue (batch_queue) and runs the provided
    batch processor function. Processed results are then stored in a shared result
    map, keyed by their respective request IDs.
    Args:
        batch_queue: A Ray Queue which stores full batches. Each item should be
            a tuple of (request_data_batch, request_id-batch).
        result_map: A shared Ray mapping (dictionary-like object) where
            processed results are stored. Keys are request IDs, and values are results.
        batch_processor: A function that takes a batch (list) of request data
            and returns a list of processed results. The function should be designed to
            optimize batch processing.
    Returns:
        None: This function runs indefinitely and returns nothing.
    """
    logging.debug("Starting batch processor process...")
    print("Starting batch processor process...")
    while True:
        try:
            batch, id_batch = batch_queue.get()
            #logging.debug("Running batch processor...")
            #print("Running batch processor...")

            results = batch_processor(batch)
            #print("Post processing results")
            for result, request_id in zip(results, id_batch):
                logging.debug("Writing end results: %s, %s", result, request_id)
                result_map.set.remote(k=request_id, v=result)
            #print("Done postprocessing batch.")
        except ray.util.queue.Empty as e:
            logging.debug("Caught exception: %s", e)


@ray.remote
class _BatcherResults:
    """A very simple implementation of a dictionary hosted in a Ray actor.
    This is primarily used to hold the final mapping between the request ID
    and the final results.
    """

    def __init__(self):
        self._d = {}

    def set(self, k: Any, v: Any):
        """Sets a key value pair within the internal dict."""
        self._d[k] = v

    def get(self, k: Any) -> Optional[Any]:
        """Gets the value based on the provided key."""
        try:
            return self._d[k]
        except KeyError:
            return None

    def print(self):
        """Prints the contents of the dictionary."""
        print(self._d)


@ray.remote
class RequestBatcher:
    """
    A handler for batching together individual requests for batch-level processing within a Ray actor.
    This class provides an efficient way to aggregate requests into larger batches for optimized computation,
    while also handling the complexities of asynchronous request and response management.

    Attributes:
        batch_handler_fn:
            The function responsible for processing a batch of requests.
            It takes an iterable of request data and returns a list containing the results of each request.
        batch_size:
            The target size of each batch of requests. The class will attempt to aggregate this many
            requests into each batch before calling `batch_handler_fn`.
        batch_timeout_s:
            The maximum amount of time, in seconds, to wait for a batch to fill up before processing it.
            If the `batch_size` is not met within this timeframe, the batch will be processed with fewer items.
        _request_queue:
            Internal queue for storing incoming request data along with their unique IDs.
        _batch_queue:
            Internal queue to hold batches of requests that are ready for processing.
        _result_map:
            Internal mapping of request IDs to results. This is used to retrieve the result of a particular request.
        _batch_task_handle:
            A handle to the internal Ray task responsible for assembling batches from the `_request_queue`.
        _batch_processor_task_handle:
            A handle to the internal Ray task responsible for processing batches from the `_batch_queue`
            and storing results in `_result_map`.
    """

    def __init__(
        self,
        batch_handler_fn: Callable[[Iterable[Any]], List[Any]],
        batch_size: int,
        batch_timeout_s: int,
    ):
        print("Creating the Request Batcher actor")
        logging.info("Creating the Request Batcher actor")
        self.batch_handler_fn = batch_handler_fn
        self.batch_size = batch_size
        self.batch_timeout_s = batch_timeout_s
        self._request_queue = Queue()
        self._batch_queue = Queue()
        self._result_map = _BatcherResults.remote()
        self._batch_task_handle = _batcher_task.remote(
            request_queue=self._request_queue,
            batch_queue=self._batch_queue,
            target_batch_size=self.batch_size,
            timeout_in_s=self.batch_timeout_s,
        )
        print("Done Creating the Request Batcher handle")
        logging.info("Done Creating the Request Batcher handle")
        self._batch_processor_task_handle = _batch_processor_task.remote(
            batch_queue=self._batch_queue,
            result_map=self._result_map,
            batch_processor=self.batch_handler_fn,
        )
        print("Done Creating the Request Batcher handle processor")
        logging.info("Done Creating the Request Batcher handle processor")

    def put(self, input: Any) -> str:
        """Adds a new request to the batcher.

        Args:
            input: The request data to be processed.
        Returns:
            A unique identifier for the request, which can be used to retrieve the result later.
        """
        request_id = uuid.uuid4()
        self._request_queue.put((input, request_id))
        return request_id

    def get(self, request_id: str) -> Any:
        """Retrieves the result for a given request ID.

        Args:
            request_id: The unique identifier for the request whose result is being retrieved.

        Returns:
            The result of the individual request, as processed by `batch_handler_fn`.
        """

        while not ray.get(self._result_map.get.remote(request_id)):
            pass
        return ray.get(self._result_map.get.remote(k=request_id))

    def exit(self):
        """Cleans up the Ray tasks and resources associated with this instance."""
        ray.cancel(self._batch_task_handle)
        ray.kill(self._request_queue)
        ray.kill(self._result_map)
