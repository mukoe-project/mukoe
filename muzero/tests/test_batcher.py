"""Tests for batching/batcher.py
Run this with:
python3 -m pytest -v -rP tests/test_batcher.py
"""
import pytest
import ray
import sys
from batching.batcher import RequestBatcher


def test_basic_usage():
    ray.init()
    batcher = RequestBatcher.remote(
        batch_handler_fn=lambda x: x, batch_size=4, batch_timeout_s=1
    )
    request_ids = ray.get([batcher.put.remote(str(i)) for i in range(4)])
    request_handles = [batcher.get.remote(request_id) for request_id in request_ids]
    results = ray.get(request_handles)
    assert results == ["0", "1", "2", "3"]
    ray.shutdown()


def test_timeout_triggers():
    """Tests that the timeout can trigger batch processing, e.g. provided batch < batch_size."""
    ray.init()
    batcher = RequestBatcher.remote(
        batch_handler_fn=lambda x: x, batch_size=4, batch_timeout_s=1
    )
    request_ids = ray.get([batcher.put.remote(str(i)) for i in range(2)])
    request_handles = [batcher.get.remote(request_id) for request_id in request_ids]
    results = ray.get(request_handles)
    assert results == ["0", "1"]
    ray.shutdown()


def test_multiple_batches():
    """Tests that we can run batching on multiple batches."""
    ray.init()
    batcher = RequestBatcher.remote(
        batch_handler_fn=lambda x: x, batch_size=2, batch_timeout_s=1
    )
    request_ids = ray.get([batcher.put.remote(str(i)) for i in range(4)])
    request_handles = [batcher.get.remote(request_id) for request_id in request_ids]
    results = ray.get(request_handles)
    assert results == ["0", "1", "2", "3"]
    ray.shutdown()


def test_full_and_half_batch():
    """Tests that we can run batching on a full batch and a half batch."""
    ray.init()
    batcher = RequestBatcher.remote(
        batch_handler_fn=lambda x: x, batch_size=2, batch_timeout_s=1
    )
    request_ids = ray.get([batcher.put.remote(str(i)) for i in range(6)])
    request_handles = [batcher.get.remote(request_id) for request_id in request_ids]
    results = ray.get(request_handles)
    assert results == ["0", "1", "2", "3", "4", "5"]
    ray.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-sv", __file__]))
