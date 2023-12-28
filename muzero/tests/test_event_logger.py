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
    owner_name = "tester"
    event_category= "testCase"
    event_id = "1"
    args = dict(
        owner_name=owner_name,
        event_category=event_category,
        event_id=event_id)
    event_logger.event_start(**args)
    time.sleep(2)
    event_logger.event_stop(**args)
    event_logger.summary()
    time.sleep(2)
    ray.shutdown()


def test_multiple_calls():
    ray.init()
    owner_name = "tester"
    event_category= "testCase"
    args = dict(
        owner_name=owner_name,
        event_category=event_category)
    for _ in range(10):
        event_logger.event_start(**args)
        time.sleep(0.5)
        event_logger.event_stop(**args)

    event_logger.summary()
    time.sleep(3)
    ray.shutdown()
