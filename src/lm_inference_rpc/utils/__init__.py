"""
Shared utilities and helpers for client-server communication.
"""

from .vlm_utils import (
    # Core encoding/decoding functions (4 main functions)
    encode_request,
    decode_request_tensors,
    encode_response,
    decode_response_tensors,
    # Media processing utilities
    prepare_media,
    pil_image_to_array,
    array_to_pil_image,
    is_sequence_of_pil_images,
    tensors_to_media,
    # Constants and types
    MediaType,
    PIL_AVAILABLE,
)

__all__ = [
    # Core encoding/decoding functions
    "encode_request",
    "decode_request_tensors",
    "encode_response",
    "decode_response_tensors",
    # Media processing utilities
    "prepare_media",
    "pil_image_to_array",
    "array_to_pil_image",
    "is_sequence_of_pil_images",
    "tensors_to_media",
    # Constants and types
    "MediaType",
    "PIL_AVAILABLE",
]

