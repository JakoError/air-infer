"""
ROS2 utilities for message serialization and deserialization.
"""
from typing import Any, Optional
import numpy as np

try:
    from rclpy.serialization import serialize_message, deserialize_message
    from rosidl_runtime_py.utilities import get_message

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    serialize_message = None
    deserialize_message = None
    get_message = None


def require_ros2() -> None:
    """
    Require ROS2 to be available.

    Raises:
        ImportError: If ROS2 is not available
    """
    if not ROS2_AVAILABLE:
        raise ImportError("ROS2 is required. Install with: pip install rclpy")


def serialize_ros_message(msg: Any) -> bytes:
    """
    Serialize a ROS2 message to bytes.

    Args:
        msg: ROS2 message object

    Returns:
        Serialized message as bytes

    Raises:
        ImportError: If ROS2 is not available
    """
    require_ros2()

    return serialize_message(msg)


def deserialize_ros_message(msg_type_str: str, data: bytes) -> Any:
    """
    Deserialize bytes to a ROS2 message using dynamic type lookup.

    Args:
        msg_type_str: ROS2 message type string (e.g., "std_msgs/String")
        data: Serialized message bytes

    Returns:
        Deserialized ROS2 message object

    Raises:
        ImportError: If ROS2 is not available
        ValueError: If message type cannot be found
    """
    require_ros2()

    msg_class = get_message(msg_type_str)
    if msg_class is None:
        raise ValueError(f"Cannot find message type: {msg_type_str}")

    return deserialize_message(data, msg_class)


def get_ros_message_type(msg: Any) -> str:
    """
    Get the ROS2 message type string from a message instance.

    Args:
        msg: ROS2 message object

    Returns:
        Message type string (e.g., "std_msgs/String")

    Raises:
        ImportError: If ROS2 is not available
        AttributeError: If message doesn't have type information
    """
    require_ros2()

    # ROS2 messages have __class__.__module__ and __class__.__name__
    # Format: "std_msgs.msg._String" -> "std_msgs/String"
    module_parts = msg.__class__.__module__.split('.')
    msg_name = msg.__class__.__name__

    # Extract package name (usually the first part before 'msg')
    package_name = module_parts[0] if len(module_parts) > 0 else 'unknown'
    for i, part in enumerate(module_parts):
        if part == 'msg':
            if i > 0:
                package_name = '.'.join(module_parts[:i])
            break

    return f"{package_name}/{msg_name}"
