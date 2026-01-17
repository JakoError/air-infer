"""
Utilities for message verification and integrity checking.
"""
import hashlib
from typing import Any, Optional, Tuple


def compute_message_checksum(message_bytes: bytes) -> str:
    """
    Compute a SHA256 checksum for message bytes.
    
    Args:
        message_bytes: Serialized message bytes
        
    Returns:
        Hexadecimal checksum string
    """
    return hashlib.sha256(message_bytes).hexdigest()


def compute_message_hash(message_bytes: bytes, algorithm: str = 'sha256') -> str:
    """
    Compute a hash for message bytes using the specified algorithm.
    
    Args:
        message_bytes: Serialized message bytes
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hexadecimal hash string
    """
    if algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'sha1':
        hasher = hashlib.sha1()
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'sha512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    hasher.update(message_bytes)
    return hasher.hexdigest()


def verify_message_integrity(
        original_bytes: bytes,
        received_checksum: str,
        algorithm: str = 'sha256'
) -> Tuple[bool, Optional[str]]:
    """
    Verify message integrity by comparing checksums.
    
    Args:
        original_bytes: Original message bytes that were sent
        received_checksum: Checksum received from server
        algorithm: Hash algorithm used ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Tuple of (is_valid, computed_checksum)
        - is_valid: True if checksums match
        - computed_checksum: The computed checksum for comparison
    """
    computed_checksum = compute_message_hash(original_bytes, algorithm)
    is_valid = computed_checksum.lower() == received_checksum.lower()
    return is_valid, computed_checksum


def verify_message_content(
        original_msg: Any,
        received_msg: Any,
        compare_fields: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Verify that two ROS2 messages have the same content.
    
    Args:
        original_msg: Original ROS2 message object
        received_msg: Received ROS2 message object
        compare_fields: If True, compare message fields; if False, only compare types
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if messages match
        - error_message: Error description if messages don't match, None otherwise
    """
    # Check message types
    if type(original_msg) != type(received_msg):
        return False, f"Message type mismatch: {type(original_msg)} vs {type(received_msg)}"

    if not compare_fields:
        return True, None

    # Compare message fields
    try:
        from rclpy.serialization import serialize_message
        orig_bytes = serialize_message(original_msg)
        recv_bytes = serialize_message(received_msg)

        if orig_bytes != recv_bytes:
            return False, "Message content does not match (serialized bytes differ)"

        return True, None
    except Exception as e:
        return False, f"Error comparing messages: {str(e)}"
