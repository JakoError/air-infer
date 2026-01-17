"""
Example ROS2 server using ROSTritonReceiver.

This example demonstrates how to receive and process ROS2 messages through Triton.
"""
import sys
import threading
import time
import argparse
import os

try:
    from std_msgs.msg import String, Int32
    from geometry_msgs.msg import Point

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 messages not available. Install ROS2 or source the workspace.")
    sys.exit(1)

from lm_inference_rpc.server import ROSTritonReceiver

# Global counter for validation
received_messages = []
received_lock = threading.Lock()

# Global flag to control printing (can be set via command line or environment)
VERBOSE_OUTPUT = True


def message_handler(message):
    """
    Handle incoming ROS2 messages.

    Args:
        message: Deserialized ROS2 message object

    Returns:
        True if message was processed successfully
    """
    global VERBOSE_OUTPUT

    with received_lock:
        msg_info = {
            "type": message.__class__.__name__,
            "data": str(message)
        }
        received_messages.append(msg_info)
        if VERBOSE_OUTPUT:
            print(f"Received: {msg_info['type']} - {msg_info['data']}")

    # Validate message content based on type
    if isinstance(message, String):
        assert hasattr(
            message, 'data'), "String message should have 'data' field"
        assert isinstance(message.data, str), "String.data should be a string"
        if VERBOSE_OUTPUT:
            print(f"  ✓ Validated String message: '{message.data}'")

    elif isinstance(message, Int32):
        assert hasattr(
            message, 'data'), "Int32 message should have 'data' field"
        assert isinstance(message.data, int), "Int32.data should be an integer"
        if VERBOSE_OUTPUT:
            print(f"  ✓ Validated Int32 message: {message.data}")

    elif isinstance(message, Point):
        assert hasattr(message, 'x'), "Point message should have 'x' field"
        assert hasattr(message, 'y'), "Point message should have 'y' field"
        assert hasattr(message, 'z'), "Point message should have 'z' field"
        assert isinstance(message.x, (int, float)
                          ), "Point.x should be a number"
        if VERBOSE_OUTPUT:
            print(
                f"  ✓ Validated Point message: ({message.x}, {message.y}, {message.z})")

    return True


def main():
    """Example usage of ROSTritonReceiver."""
    global VERBOSE_OUTPUT

    parser = argparse.ArgumentParser(
        description="ROS2 Triton receiver server example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--silent', '-s',
        action='store_true',
        help='Disable printing of received message data (useful for performance testing)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server host address (default: 127.0.0.1)'
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        default=9100,
        help='Server gRPC port (default: 9100)'
    )

    parser.add_argument(
        '-m', '--model-name',
        type=str,
        default='ROSMessageHandler',
        help='Model name (default: ROSMessageHandler)'
    )

    parser.add_argument(
        '--enable-verification',
        action='store_true',
        help='Enable message checksum verification (returns checksum in response)'
    )

    args = parser.parse_args()

    # Check environment variable or command line argument
    if args.silent or os.getenv('ROS_RECEIVER_SILENT', '').lower() in ('1', 'true', 'yes'):
        VERBOSE_OUTPUT = False

    print("Creating ROS2 Triton server...")
    server = ROSTritonReceiver(
        model_name=args.model_name,
        inference_func=message_handler,
        host=args.host,
        grpc_port=args.port,
        log_verbose=0,
        enable_verification=args.enable_verification,
    )

    print("Starting server...")
    print("Server will listen on gRPC {}:{}".format(args.host, args.port))
    if VERBOSE_OUTPUT:
        print("Message printing: ENABLED")
    else:
        print("Message printing: DISABLED (silent mode)")
    if args.enable_verification:
        print("Message verification: ENABLED (checksums will be returned)")
    else:
        print("Message verification: DISABLED")
    print("Press Ctrl+C to stop...\n")

    try:
        # Note: start() blocks, so run in a separate thread for testing
        # In production, you can just call start() directly
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(2)

        print("Server is running. Waiting for messages...")
        if not VERBOSE_OUTPUT:
            print("(Silent mode: received messages will not be printed)\n")

        # Keep running until interrupted
        last_count = 0
        while True:
            time.sleep(1)
            with received_lock:
                count = len(received_messages)
            if count != last_count:
                if not VERBOSE_OUTPUT:
                    # Only print count updates in silent mode (verbose mode shows each message)
                    print(f"Total messages received: {count}")
                last_count = count

    except KeyboardInterrupt:
        print("\n\nStopping server...")
        server.stop()
        print(f"Total messages processed: {len(received_messages)}")
        print("Server stopped.")


if __name__ == "__main__":
    main()
