"""
Example server usage using TritonServer.

This example shows how to use TritonServer with an inference function
that processes media (images, videos, URLs) and JSON arguments.
"""
from air_infer.server import VLMTritonServer


def my_inference_function(media=None, **kwargs):
    """
    Example inference function that processes media and arguments.
    
    This function receives:
      - media: a list of media items (images, videos, URLs) reconstructed on the server side
      - kwargs: all additional arguments unpacked from ARGS_JSON
    
    Args:
        media: List of media items [item1, item2, ...] where each item is:
            - PIL.Image.Image (image)
            - List of PIL.Image.Image (video frames)
            - URL string
        **kwargs: Additional arguments unpacked from ARGS_JSON (e.g., prompt, temperature, etc.)
    
    Returns:
        High-level Python object (e.g., dict). It will be encoded to tensors by the server.
    """
    media = media or []

    # Example: derive some simple metadata
    num_items = len(media)
    num_images = sum(1 for m in media if not isinstance(m, (list, tuple)) and not isinstance(m, str))
    num_videos = sum(1 for m in media if isinstance(m, (list, tuple)))
    num_urls = sum(1 for m in media if isinstance(m, str))

    # Prepare results
    results = {
        "num_items": num_items,
        "num_images": num_images,
        "num_videos": num_videos,
        "num_urls": num_urls,
        "kwargs": kwargs,
        "status": "success",
    }

    # Return high-level result; server will encode it to RESULTS_JSON
    return results


def main():
    """Example usage of TritonServer with an inference function."""
    # Create server with your inference function
    server = VLMTritonServer(
        model_name="MyVLM",
        inference_func=my_inference_function,  # Pass your function here
        host="127.0.0.1",
        grpc_port=9100,
        http_port=8100,
        metrics_port=8101,
        log_verbose=0,
    )

    # Start serving (blocks until interrupted)
    print("Starting server...")

    # Use context manager (recommended)
    with server as triton:
        triton.serve()  # This will block and serve requests until interrupted

    # Or use start() method directly
    # server.start()  # This blocks


if __name__ == "__main__":
    main()
