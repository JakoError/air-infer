"""
Example demonstrating how to bind multiple models to the same Triton instance.

When multiple servers use the same host and port configuration, they will
automatically share the same Triton instance. Each server binds its own model,
but only one server needs to call serve().
"""
import threading
import time
from air_infer.server import VLMTritonServer, ROSTritonReceiver


def model1_inference(media=None, **kwargs):
    """Inference function for Model 1."""
    return {"result": "Model1", "media_count": len(media) if media else 0}


def model2_inference(media=None, **kwargs):
    """Inference function for Model 2."""
    return {"result": "Model2", "media_count": len(media) if media else 0}


def ros_handler(message):
    """ROS message handler."""
    print(f"Received ROS message: {message}")
    return True


def main():
    """
    Example: Multiple models sharing the same Triton instance.
    
    Both servers use the same host and ports, so they will share
    the same Triton instance. Only one needs to call serve().
    """
    # Create two VLM servers with the same configuration
    # They will share the same Triton instance
    server1 = VLMTritonServer(
        model_name="Model1",
        inference_func=model1_inference,
        host="127.0.0.1",
        grpc_port=9100,
        http_port=8100,
        metrics_port=8101,
    )
    
    server2 = VLMTritonServer(
        model_name="Model2",
        inference_func=model2_inference,
        host="127.0.0.1",  # Same host
        grpc_port=9100,    # Same ports
        http_port=8100,
        metrics_port=8101,
    )
    
    # Create a ROS receiver with the same configuration
    # It will also share the same Triton instance
    ros_server = ROSTritonReceiver(
        model_name="ROSHandler",
        inference_func=ros_handler,
        host="127.0.0.1",  # Same host
        grpc_port=9100,    # Same ports
        http_port=8100,
        metrics_port=8101,
    )
    
    print("=" * 60)
    print("Example: Multiple models on same Triton instance")
    print("=" * 60)
    print("\nAll three servers will share the same Triton instance")
    print("because they use the same host and port configuration.")
    print("\nOnly one server needs to call serve() - the others will")
    print("bind their models and return immediately.\n")
    
    # Start all servers in separate threads
    # Only the first one to call start() will actually serve
    def start_server1():
        print("Starting server1...")
        server1.start()  # This will block and serve
    
    def start_server2():
        time.sleep(0.5)  # Give server1 time to start
        print("Starting server2...")
        server2.start()  # This will bind and return (already serving)
    
    def start_ros_server():
        time.sleep(0.5)  # Give server1 time to start
        print("Starting ROS server...")
        ros_server.start()  # This will bind and return (already serving)
    
    # Start server1 in a thread (it will serve)
    thread1 = threading.Thread(target=start_server1, daemon=True)
    thread1.start()
    
    # Give it a moment to initialize
    time.sleep(1)
    
    # Start server2 and ros_server (they will bind but not serve)
    thread2 = threading.Thread(target=start_server2, daemon=True)
    thread2.start()
    
    thread3 = threading.Thread(target=start_ros_server, daemon=True)
    thread3.start()
    
    # Wait a bit to see the output
    time.sleep(2)
    
    print("\n" + "=" * 60)
    print("All models are now bound to the same Triton instance!")
    print("=" * 60)
    print("\nYou can now send requests to:")
    print("  - Model1: model_name='Model1'")
    print("  - Model2: model_name='Model2'")
    print("  - ROSHandler: model_name='ROSHandler'")
    print("\nAll on the same Triton instance at 127.0.0.1:9100")
    print("\nPress Ctrl+C to stop...")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping servers...")
        server1.stop()
        server2.stop()
        ros_server.stop()
        print("Servers stopped.")


if __name__ == "__main__":
    main()
