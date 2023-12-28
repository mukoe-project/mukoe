"""Tests for event_logger.py.

Run this with:
python3 -m pytest -v -rP tests/test_event_logger.py
"""

import pytest
import ray
import event_logger
import time


def test_basic_usage():
    ray.init()
    actor_name = "testActor"
    event_category= "testCase"
    event_id = "1"
    args = dict(
        actor_name=actor_name,
        event_category=event_category,
        event_id=event_id)
    event_logger.event_start(**args)
    time.sleep(5)
    event_logger.event_stop(**args)
    event_logger.summary()
    time.sleep(3)
    ray.shutdown()


def test_multiple_calls():
    ray.init()
    actor_name = "testActor"
    event_category= "testCase"
    args = dict(
        actor_name=actor_name,
        event_category=event_category)
    for _ in range(10):
        event_logger.event_start(**args)
        time.sleep(1)
        event_logger.event_stop(**args)

    event_logger.summary()
    time.sleep(3)
    ray.shutdown()
