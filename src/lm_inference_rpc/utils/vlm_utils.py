"""
VLM utilities for encoding/decoding between high-level Python objects and tensors.

This module provides 4 core encoding/decoding functions for client-server communication:

1. encode_request (client side):
   - Converts high-level inputs (media list + kwargs) → Tensors
   - Used by client to prepare requests for the server

2. decode_request_tensors (server side):
   - Converts Tensors → High-level inputs (media list + kwargs)
   - Used by server to reconstruct user-friendly inputs

3. encode_response (server side):
   - Converts high-level output (Python object) → Tensors
   - Used by server to prepare responses for transmission

4. decode_response_tensors (client side):
   - Converts Tensors → High-level output (Python object)
   - Used by client to process server responses

Additional helper functions are provided for media processing.
"""
from enum import IntEnum
from typing import List, Tuple, Optional, Union, Sequence, Dict, Any
import numpy as np
import json

try:
    from PIL import Image as PIL_Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PIL_Image = None


def is_sequence_of_pil_images(x) -> bool:
    """Check if x is a sequence of PIL Images."""
    if not PIL_AVAILABLE:
        return False
    if isinstance(x, (str, bytes)):
        return False
    if isinstance(x, Sequence):
        if len(x) == 0:
            return False
        return all(isinstance(e, PIL_Image.Image) for e in x)
    return False


class MediaType(IntEnum):
    """Media type identifiers used in MEDIA_MASK."""

    IMAGE = 0
    VIDEO = 1
    URL = 2


def _is_url(s: str) -> bool:
    """Check if a string looks like a URL."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    return s.startswith(('http://', 'https://', 'ftp://', 'file://', 's3://', 'gs://'))


def prepare_media(media) -> List[Tuple[str, Union[PIL_Image.Image, Sequence, str], Optional[dict]]]:
    """
    Prepare media from various formats, supporting multiple media items.

    This function handles PIL Images, sequences of PIL Images, and URLs.

    Args:
        media: List of media items [item1, item2, ...] where each item can be:
              - PIL.Image.Image (one image)
              - Sequence of PIL.Image (one video - sequence of frames)
              - String URL (http://, https://, etc.) (one URL)
              - Tuple: (media_item, media_info) or (media_item,)
              - Dict: {'media': media_item, 'media_info': media_info} or {'media': media_item, 'fps': fps, ...}
              Note: media should be a list. A single item will be wrapped in a list for convenience.

    Returns:
        list: List of tuples [(media_type, media_value, media_info), ...] where:
            - media_type: "image", "video", or "url"
            - media_value: PIL.Image, list of PIL.Images, or URL string
            - media_info: dict with metadata (e.g., {'fps': fps}) or None
    """
    # Convert single item to list for backward compatibility
    if not isinstance(media, Sequence) or isinstance(media, str):
        media_list = [media] if media is not None else []
    else:
        media_list = media

    if len(media_list) == 0:
        return []

    results = []

    # Process each media item
    for media_item in media_list:
        # Extract media and metadata from tuple or dict format
        media_info = {}
        if isinstance(media_item, tuple):
            actual_media = media_item[0]
            if len(media_item) > 1:
                if isinstance(media_item[1], dict):
                    media_info = media_item[1]
        elif isinstance(media_item, dict):
            actual_media = media_item.get('media', None)
            for key in ['fps', 'video_fps', 'video_metadata', 'media_info']:
                if key in media_item:
                    if key == 'media_info' and isinstance(media_item[key], dict):
                        media_info.update(media_item[key])
                    elif key == 'video_metadata' and isinstance(media_item[key], dict):
                        media_info.update(media_item[key])
                    elif key in ['fps', 'video_fps']:
                        media_info['fps'] = media_item[key]
        else:
            actual_media = media_item

        if actual_media is None:
            continue

        # Normalize media_info
        if 'video_fps' in media_info and 'fps' not in media_info:
            media_info['fps'] = media_info.pop('video_fps')

        # Determine media type and value
        if isinstance(actual_media, str) and _is_url(actual_media):
            # URL string
            results.append(
                ("url", actual_media, media_info if media_info else None))
        elif PIL_AVAILABLE and isinstance(actual_media, PIL_Image.Image):
            results.append(
                ("image", actual_media, media_info if media_info else None))
        elif PIL_AVAILABLE and is_sequence_of_pil_images(actual_media):
            results.append(
                ("video", actual_media, media_info if media_info else None))
        elif isinstance(actual_media, str):
            # String that's not a URL - treat as URL anyway (could be file path)
            results.append(
                ("url", actual_media, media_info if media_info else None))
        else:
            # Unknown type - skip
            continue

    return results


def pil_image_to_array(img: PIL_Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (uint8, RGB)."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required")
    # Ensure RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.asarray(img, dtype=np.uint8)


def array_to_pil_image(arr: np.ndarray) -> PIL_Image.Image:
    """Convert numpy array to PIL Image."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required")
    return PIL_Image.fromarray(arr.astype(np.uint8), mode='RGB')


def encode_request(
        media: Optional[Union[List, Sequence, Any]] = None,
        **kwargs
) -> Dict[str, np.ndarray]:
    """
    Encode high-level request (media + kwargs) into input tensors for the server.

    This is one of the 4 core encoding/decoding functions:
    - encode_request: High-level inputs → Tensors (client side)
    - decode_request_tensors: Tensors → High-level inputs (server side)
    - encode_response: High-level output → Tensors (server side)
    - decode_response_tensors: Tensors → High-level output (client side)

    This function processes media items (PIL Images, sequences of PIL Images, URLs)
    and converts them to the format expected by the server. All other keyword arguments
    are packed into ARGS_JSON.

    Args:
        media: List of media items [item1, item2, ...] where each item can be:
            - PIL.Image.Image (one image item)
            - Sequence of PIL.Image (one video item - sequence of frames)
            - String URL (http://, https://, etc.) (one URL item)
            - Tuple: (media_item, media_info) or (media_item,)
            - Dict: {'media': media_item, 'fps': fps, ...}
            Note: A single item will be automatically wrapped in a list.
        **kwargs: All other arguments will be packed into a dict and converted to JSON

    Returns:
        Dictionary of input tensors ready for inference with keys:
        - IMAGE: Concatenated image array (N_IMAGES, H, W, C) or empty array
        - VIDEO: Concatenated video array (N_VIDEOS, T, H, W, C) or empty array
        - MEDIA_URLS: URLs as bytes tensor (N_URLS,) or empty array
        - MEDIA_MASK: Mask indicating media type order (MediaType.IMAGE, MediaType.VIDEO, MediaType.URL)
        - ARGS_JSON: JSON string containing all kwargs as bytes tensor
    """
    inputs = {}

    # Process media items
    image_arrays = []
    video_arrays = []
    url_list = []
    media_mask = []

    if media is not None:
        # Prepare media using utility function (handles PIL Images, videos, and URLs)
        prepared_media = prepare_media(media)

        for media_type, media_value, media_info in prepared_media:
            if media_type == "image":
                if not PIL_AVAILABLE:
                    raise ImportError(
                        "PIL (Pillow) is required for image processing. Install with: pip install Pillow")
                # Convert PIL Image to numpy array
                img_array = pil_image_to_array(media_value)
                image_arrays.append(img_array)
                media_mask.append(MediaType.IMAGE)  # 0 for image
            elif media_type == "video":
                if not PIL_AVAILABLE:
                    raise ImportError(
                        "PIL (Pillow) is required for video processing. Install with: pip install Pillow")
                # Convert sequence of PIL Images to numpy array
                # Shape: (T, H, W, C)
                video_frames = [pil_image_to_array(
                    frame) for frame in media_value]
                video_array = np.stack(video_frames, axis=0)  # (T, H, W, C)
                video_arrays.append(video_array)
                media_mask.append(MediaType.VIDEO)  # 1 for video
            elif media_type == "url":
                # URL string
                url_list.append(media_value)
                media_mask.append(MediaType.URL)  # 2 for URL

    # Concatenate images if any
    if len(image_arrays) > 0:
        # Stack images: (N_IMAGES, H, W, C) where N_IMAGES = number of images
        inputs["IMAGE"] = np.stack(image_arrays, axis=0)
    else:
        inputs["IMAGE"] = np.zeros((1, 1, 1, 3), dtype=np.uint8)  # Empty array

    # Concatenate videos if any
    if len(video_arrays) > 0:
        # Stack videos: each is (T, H, W, C), stack to (N_VIDEOS, T, H, W, C)
        # Videos may have different T, H, W, so pad to max dimensions
        if len(video_arrays) == 1:
            # Single video, no padding needed
            inputs["VIDEO"] = np.expand_dims(
                video_arrays[0], axis=0)  # (1, T, H, W, C)
        else:
            max_t = max(v.shape[0] for v in video_arrays)
            max_h = max(v.shape[1] for v in video_arrays)
            max_w = max(v.shape[2] for v in video_arrays)
            c = video_arrays[0].shape[3]  # All should have same channels

            padded_videos = []
            for v in video_arrays:
                t, h, w, c_v = v.shape
                if c_v != c:
                    raise ValueError(
                        f"All videos must have the same number of channels. Got {c} and {c_v}")
                if t != max_t or h != max_h or w != max_w:
                    # Pad video to max dimensions
                    padded = np.zeros((max_t, max_h, max_w, c), dtype=np.uint8)
                    padded[:t, :h, :w, :] = v
                    padded_videos.append(padded)
                else:
                    padded_videos.append(v)

            inputs["VIDEO"] = np.stack(padded_videos, axis=0)  # (N_VIDEOS, T, H, W, C)
    else:
        inputs["VIDEO"] = np.zeros(
            (1, 1, 1, 1, 3), dtype=np.uint8)  # Empty array

    # Process URLs
    if len(url_list) > 0:
        # Convert URLs to bytes tensor
        url_bytes = np.array([url.encode('utf-8')
                              for url in url_list], dtype=np.object_)
        inputs["MEDIA_URLS"] = url_bytes
    else:
        # Empty array with shape (0,)
        inputs["MEDIA_URLS"] = np.array([''.encode('utf-8')], dtype=np.object_)

    # Create media mask
    if len(media_mask) > 0:
        inputs["MEDIA_MASK"] = np.array(media_mask, dtype=np.uint8)
    else:
        inputs["MEDIA_MASK"] = np.array([], dtype=np.uint8)

    # Pack all kwargs into JSON
    if kwargs:
        # Convert kwargs to JSON string
        args_json_str = json.dumps(kwargs)
        args_json_bytes = args_json_str.encode('utf-8')
    else:
        args_json_bytes = b'{}'  # Default empty JSON

    inputs["ARGS_JSON"] = np.array([args_json_bytes], dtype=np.object_)

    return inputs


def tensors_to_media(
        IMAGE: Optional[np.ndarray] = None,
        VIDEO: Optional[np.ndarray] = None,
        MEDIA_URLS: Optional[np.ndarray] = None,
        MEDIA_MASK: Optional[np.ndarray] = None,
) -> List[Tuple[str, Union[Any, str], Optional[dict]]]:
    """
    Reconstruct media items from input tensors using MEDIA_MASK.

    This function splits the concatenated IMAGE and VIDEO arrays back into
    individual media items and reconstructs URLs, following the order
    specified in MEDIA_MASK.

    Note: This function only handles inputs without batch dimension.
    IMAGE must be (N_IMAGES, H, W, C) and VIDEO must be (N_VIDEOS, T, H, W, C).

    Args:
        IMAGE: Concatenated image array (N_IMAGES, H, W, C) or None
        VIDEO: Concatenated video array (N_VIDEOS, T, H, W, C) or None
        MEDIA_URLS: URLs as bytes tensor (N_URLS,) or None
        MEDIA_MASK: Mask indicating media type order (N_IMAGES + N_VIDEOS + N_URLS)

    Returns:
        List of tuples [(media_type, media_value, media_info), ...] where:
        - media_type: "image", "video", or "url"
        - media_value: PIL.Image, list of PIL.Images, or URL string
        - media_info: dict with metadata or None
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for media reconstruction")

    if MEDIA_MASK is None or len(MEDIA_MASK) == 0:
        return []

    results = []
    image_idx = 0
    video_idx = 0
    url_idx = 0

    # Check if IMAGE/VIDEO are empty or have proper shape
    # IMAGE: (N_IMAGES, H, W, C) - no batch dimension
    # VIDEO: (N_VIDEOS, T, H, W, C) - no batch dimension
    has_images = False
    image_shape = None
    if IMAGE is not None:
        if len(IMAGE.shape) != 4:
            raise ValueError(
                f"IMAGE must have shape (N_IMAGES, H, W, C), got shape {IMAGE.shape}")
        has_images = IMAGE.shape[0] > 0
        image_shape = IMAGE.shape

    has_videos = False
    video_shape = None
    if VIDEO is not None:
        if len(VIDEO.shape) != 5:
            raise ValueError(
                f"VIDEO must have shape (N_VIDEOS, T, H, W, C), got shape {VIDEO.shape}")
        has_videos = VIDEO.shape[0] > 0
        video_shape = VIDEO.shape

    has_urls = MEDIA_URLS is not None and len(MEDIA_URLS) > 0

    for mask_value in MEDIA_MASK:
        if mask_value == MediaType.IMAGE:  # Image
            if has_images and image_idx < image_shape[0]:
                img_array = IMAGE[image_idx]  # (H, W, C)
                pil_image = array_to_pil_image(img_array)
                results.append(("image", pil_image, None))
                image_idx += 1
            else:
                raise ValueError(
                    f"MEDIA_MASK indicates image at position {len(results)}, but IMAGE array is missing or insufficient (idx={image_idx}, shape={IMAGE.shape if IMAGE is not None else None})")

        elif mask_value == MediaType.VIDEO:  # Video
            if has_videos and video_idx < video_shape[0]:
                video_array = VIDEO[video_idx]  # (T, H, W, C)
                # Convert each frame to PIL Image
                video_frames = [array_to_pil_image(
                    video_array[t]) for t in range(video_array.shape[0])]
                results.append(("video", video_frames, None))
                video_idx += 1
            else:
                raise ValueError(
                    f"MEDIA_MASK indicates video at position {len(results)}, but VIDEO array is missing or insufficient (idx={video_idx}, shape={VIDEO.shape if VIDEO is not None else None})")

        elif mask_value == MediaType.URL:  # URL
            if has_urls and url_idx < len(MEDIA_URLS):
                url_bytes = MEDIA_URLS[url_idx]
                if isinstance(url_bytes, np.ndarray):
                    url_str = url_bytes.item().decode('utf-8')
                else:
                    url_str = url_bytes.decode('utf-8')
                results.append(("url", url_str, None))
                url_idx += 1
            else:
                raise ValueError(
                    f"MEDIA_MASK indicates URL at position {len(results)}, but MEDIA_URLS is missing or insufficient (idx={url_idx}, len={len(MEDIA_URLS) if MEDIA_URLS is not None else 0})")

        else:
            raise ValueError(f"Unknown MEDIA_MASK value: {mask_value}")

    return results


def decode_request_tensors(
        IMAGE: Optional[np.ndarray] = None,
        VIDEO: Optional[np.ndarray] = None,
        MEDIA_URLS: Optional[np.ndarray] = None,
        MEDIA_MASK: Optional[np.ndarray] = None,
        ARGS_JSON: Optional[np.ndarray] = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Decode raw input tensors into high-level media list and kwargs.

    This is one of the 4 core encoding/decoding functions:
    - encode_request: High-level inputs → Tensors (client side)
    - decode_request_tensors: Tensors → High-level inputs (server side)
    - encode_response: High-level output → Tensors (server side)
    - decode_response_tensors: Tensors → High-level output (client side)

    This function is the inverse of encode_request(). It reconstructs media items
    from tensors and decodes JSON arguments.

    Args:
        IMAGE: Concatenated image array (N_IMAGES, H, W, C) or None
        VIDEO: Concatenated video array (N_VIDEOS, T, H, W, C) or None
        MEDIA_URLS: URLs as bytes tensor (N_URLS,) or None
        MEDIA_MASK: Mask indicating media type order (MediaType.IMAGE, MediaType.VIDEO, MediaType.URL)
        ARGS_JSON: JSON arguments as bytes tensor (1,) or None

    Returns:
        Tuple of (media, kwargs):
        - media: List of media items [item1, item2, ...] where each item is:
            - PIL.Image.Image (image)
            - List[PIL.Image.Image] (video frames)
            - URL string
        - kwargs: Dict of additional arguments decoded from ARGS_JSON
    """
    # Reconstruct typed media tuples
    typed_media = tensors_to_media(
        IMAGE=IMAGE,
        VIDEO=VIDEO,
        MEDIA_URLS=MEDIA_URLS,
        MEDIA_MASK=MEDIA_MASK,
    )

    # Strip type/info, keep only the media value for user code
    media: List[Any] = [value for _type, value, _info in typed_media]

    # Decode ARGS_JSON into kwargs
    kwargs: Dict[str, Any] = {}
    if ARGS_JSON is not None:
        try:
            if isinstance(ARGS_JSON, np.ndarray) and ARGS_JSON.size > 0:
                first = ARGS_JSON[0]
                if isinstance(first, np.ndarray):
                    raw = first.item()
                else:
                    raw = first
                if isinstance(raw, (bytes, bytearray)):
                    kwargs = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"Error decoding ARGS_JSON: {e}")

    return media, kwargs


def encode_response(payload: Any) -> Dict[str, np.ndarray]:
    """
    Encode high-level response payload into output tensors for server transmission.

    This is one of the 4 core encoding/decoding functions:
    - encode_request: High-level inputs → Tensors (client side)
    - decode_request_tensors: Tensors → High-level inputs (server side)
    - encode_response: High-level output → Tensors (server side)
    - decode_response_tensors: Tensors → High-level output (client side)

    Default implementation:
      - If payload is already a dict of numpy arrays, return as-is.
      - Otherwise, JSON-encode the payload into a single RESULTS_JSON tensor.

    Args:
        payload: High-level result (e.g., dict) from user inference function.

    Returns:    
        Dict[str, np.ndarray]: Tensors matching the default output schema.
            Default: {"RESULTS_JSON": ...} with JSON-encoded bytes.
    """
    if not isinstance(payload, dict):
        payload = {"result": payload}

    data_bytes = json.dumps(payload).encode("utf-8")

    return {
        "RESULTS_JSON": np.array([data_bytes], dtype=np.object_),
    }


def decode_response_tensors(outputs: Dict[str, Any]) -> Any:
    """
    Decode server response tensors into a high-level Python object.

    This is one of the 4 core encoding/decoding functions:
    - encode_request: High-level inputs → Tensors (client side)
    - decode_request_tensors: Tensors → High-level inputs (server side)
    - encode_response: High-level output → Tensors (server side)
    - decode_response_tensors: Tensors → High-level output (client side)

    This function is the inverse of encode_response(). It decodes tensors
    back into high-level Python objects.

    Default implementation:
      - If RESULTS_JSON exists, JSON-decode it.
      - Otherwise, return outputs as-is.

    Args:
        outputs: Raw output tensors from server.

    Returns:
        Decoded Python object (usually a dict). If RESULTS_JSON is present,
        returns the JSON-decoded content. Otherwise returns outputs as-is.
    """
    if "RESULTS_JSON" not in outputs:
        return outputs

    arr = outputs["RESULTS_JSON"]
    if isinstance(arr, np.ndarray) and arr.size > 0:
        first = arr[0]
        if isinstance(first, np.ndarray):
            raw = first.item()
        else:
            raw = first
        if isinstance(raw, (bytes, bytearray)):
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                # Fallback: return raw string
                return raw.decode("utf-8", errors="replace")

    return outputs
