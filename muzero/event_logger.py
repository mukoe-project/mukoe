"""A custom logger that tracks timestamps of various events.

event_logger.event_start(
    actor_name=str(self),
    event_category=<name-of-function>,
    event_id=<>,
)
event_logger.event_stop()

"""
import ray
import time
import numpy as np
import logging
import dataclasses
from collections import defaultdict
from typing import List, Mapping, Optional


@dataclasses.dataclass
class Event:
    start: float
    stop: Optional[float] = None

# Using a nested dictionary structure for simplicity
History = Mapping[str, Mapping[str, Event]]  # actor_name -> event_category -> event_id -> Event


@ray.remote(num_cpus=1)
class EventLogger:
    def __init__(self):
        self.history: History = defaultdict(lambda: defaultdict(dict))

    def event_start(self, actor_name: str, event_category: str, timestamp: float, event_id: Optional[str] = None):
        """Records the start of an event.

        Args:
            actor_name: Name of the actor.
            event_category: Category of the event.
            timestamp: The start time of the event.
            event_id: Optional unique identifier for the event. If None,
                an ID will be generated.
        """
        if event_id is None:
            event_id = self.generate_event_id(actor_name, event_category)
        self.history[actor_name][event_category][event_id] = Event(start=timestamp)

    def event_stop(self, actor_name: str, event_category: str, timestamp: float, event_id: Optional[str] = None):
        """Records the end of an event.

        Args:
            actor_name: Name of the actor.
            event_category: Category of the event.
            timestamp: The start time of the event.
            event_id: Optional unique identifier for the event. If None,
                an ID will be generated.
        """
        if event_id is None:
            event_id = self.get_latest_event_id(actor_name, event_category)
        if event_id and event_id in self.history[actor_name][event_category]:
            self.history[actor_name][event_category][event_id].stop = timestamp

    def generate_event_id(self, actor_name: str, event_category: str) -> str:
        """
        Generates a unique event ID based on the actor name, event category, and the number of existing events.

        Args:
            actor_name: Name of the actor.
            event_category: Category of the event.

        Returns:
            Generated unique event ID.
        """
        return f"{actor_name}_{event_category}_{len(self.history[actor_name][event_category])}"

    def get_latest_event_id(self, actor_name: str, event_category: str) -> Optional[str]:
        """
        Retrieves the latest event ID for the specified actor and category.

        Args:
            actor_name: Name of the actor.
            event_category: Category of the event.

        Returns:
            The latest event ID or None if no events exist for the actor-category combination.
        """
        if self.history[actor_name][event_category]:
            return list(self.history[actor_name][event_category].keys())[-1]
        return None

    def summary(self):
        for actor, actor_history in self.history.items():
            actor_summary = {}
            print(f"Actor::{actor}")
            for event_category, event_history in actor_history.items():
                event_durations = []
                for event_id, event in event_history.items():
                    event_durations.append(event.stop - event.start)
                print(f"EventCategory::{event_category}")
                print(f"AvgTime::{np.mean(event_durations)}")
                print(f"MedianTime::{np.median(event_durations)}")
                print(f"Raw:{event_durations}")
                print(f"Num invocations: {len(event_history)}")
            print("------------------")


def get_global_logger() -> ray.actor.ActorHandle:
    t = time.perf_counter()
    try:
        global_logger = ray.get_actor("event_logger", namespace="metrics")
    except ValueError as e:
        global_logger = EventLogger.options(name="event_logger", namespace="metrics", lifetime="detached").remote()
    return global_logger


def event_start(
    actor_name: str,
    event_category: str,
    event_id: Optional[int] = None):
    #t = time.process_time()
    t = time.perf_counter()
    global_logger = get_global_logger()
    global_logger.event_start.remote(
        timestamp=t,
        actor_name=actor_name,
        event_category=event_category,
        event_id=event_id)


def event_stop(
    actor_name: str,
    event_category: str,
    event_id: Optional[int] = None):
    #t = time.process_time()
    t = time.perf_counter()
    global_logger = get_global_logger()
    global_logger.event_stop.remote(
        timestamp=t,
        actor_name=actor_name,
        event_category=event_category,
        event_id=event_id)


def summary():
    global_logger = get_global_logger()
    global_logger.summary.remote()
