"""Shared ROS 2 plumbing for the SuperMap nodes.

All nodes are managed (lifecycle) nodes so ``nav2_lifecycle_manager`` can drive
them. With the ``autostart`` parameter (default true) a node configures and
activates itself once the executor spins, so plain ``ros2 run`` / ``Node(...)``
launches keep working.
"""
from __future__ import annotations

import colorsys
import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import rclpy
from rcl_interfaces.msg import FloatingPointRange, IntegerRange, ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

__all__ = [
    "AutostartLifecycleNode",
    "TransitionCallbackReturn",
    "declare",
    "run_node",
    "stable_label_color",
]


def stable_label_color(label: str, saturation: float = 0.65, value: float = 0.95) -> tuple[float, float, float]:
    """RGB in [0, 1] derived from a process-independent hash of ``label``.

    ``hash(str)`` is salted per process (PYTHONHASHSEED), so it cannot be used
    for colours that must match across restarts and across nodes.
    """
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "little") / 2**32
    return colorsys.hsv_to_rgb(hue, saturation, value)


def declare(node, name: str, default: Any, description: str = "", *, read_only: bool = True,
            range: tuple[float, float] | None = None, step: float | None = None,
            dynamic_typing: bool = False) -> Any:
    """Declare a parameter with a descriptor and return its value.

    Most SuperMap parameters are read once while configuring, so they default
    to ``read_only`` to make ``ros2 param set`` fail loudly instead of
    silently doing nothing. Float parameters accept integer YAML values.
    """
    descriptor = ParameterDescriptor(description=description, read_only=read_only,
                                     dynamic_typing=dynamic_typing)
    if range is not None:
        if isinstance(default, bool):
            raise TypeError(f"range given for boolean parameter {name}")
        if isinstance(default, int):
            descriptor.integer_range = [IntegerRange(from_value=int(range[0]), to_value=int(range[1]),
                                                     step=int(step or 0))]
        else:
            descriptor.floating_point_range = [FloatingPointRange(from_value=float(range[0]),
                                                                  to_value=float(range[1]),
                                                                  step=float(step or 0.0))]
    if isinstance(default, float) and not dynamic_typing:
        # Accept `5` for a double parameter: declare dynamically typed and coerce.
        descriptor.dynamic_typing = True
        value = node.declare_parameter(name, default, descriptor).value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"parameter {name} must be a number, got {value!r}")
        return float(value)
    return node.declare_parameter(name, default, descriptor).value


class AutostartLifecycleNode(LifecycleNode):
    """Lifecycle node that can drive itself to ACTIVE.

    Subclasses implement ``on_configure``/``on_activate``/``on_deactivate``/
    ``on_cleanup``/``on_shutdown`` as usual and call ``super()`` for the
    activate/deactivate hooks so lifecycle publishers follow the state.
    """

    def __init__(self, node_name: str, **kwargs) -> None:
        super().__init__(node_name, **kwargs)
        self._autostart = bool(declare(
            self, "autostart", True,
            "Configure and activate on startup. Set false when nav2_lifecycle_manager manages this node."))
        self._autostart_timer = None
        if self._autostart:
            self._autostart_timer = self.create_timer(0.0, self._run_autostart)

    def _run_autostart(self) -> None:
        self.destroy_timer(self._autostart_timer)
        self._autostart_timer = None
        if self._state_machine.current_state[1] != "unconfigured":
            return  # already driven externally (lifecycle manager, tests)
        if self.trigger_configure() != TransitionCallbackReturn.SUCCESS:
            self.get_logger().error("autostart: configure failed; staying unconfigured")
            return
        if self.trigger_activate() != TransitionCallbackReturn.SUCCESS:
            self.get_logger().error("autostart: activate failed; staying inactive")

    def autostart_now(self) -> None:
        """Run the autostart transitions synchronously (tests, embedding)."""
        if self._autostart_timer is not None:
            self._run_autostart()


def run_node(factory: Callable[[], Any], args: Sequence[str] | None = None, *,
             multithreaded: bool = False, num_threads: int | None = None) -> None:
    """Standard ``main`` body: spin until Ctrl-C / ``ros2 launch`` shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = factory()
        if multithreaded:
            executor = MultiThreadedExecutor(num_threads=num_threads)
            executor.add_node(node)
            executor.spin()
        else:
            rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except KeyboardInterrupt:  # a repeated Ctrl-C while workers are joining
                pass
        rclpy.try_shutdown()
