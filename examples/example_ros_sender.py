"""
Example ROS2 client using ROSTritonSender.

This example demonstrates how to send ROS2 messages through Triton.
"""
import sys

try:
    from std_msgs.msg import String, Int32
    from geometry_msgs.msg import Point

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 messages not available. Install ROS2 or source the workspace.")
    sys.exit(1)

from lm_inference_rpc.client import ROSTritonSender


def main():
    """Example usage of ROSTritonSender."""
    import argparse

    parser = argparse.ArgumentParser(description="ROS2 Triton sender example")
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Enable message verification (requires server to have --enable-verification)'
    )
    args = parser.parse_args()

    print("Creating ROS2 Triton client...")
    client = ROSTritonSender(
        model_name="ROSMessageHandler",
        host="127.0.0.1",
        grpc_port=9100,
    )

    try:
        with client:
            print("Client connected!")
            if args.verify:
                print("Verification: ENABLED\n")

            # Example 1: Send a String message
            print("\n--- Example 1: String message ---")
            msg1 = String()
            msg1.data = "Hello, ROS2!"
            print(f"Sending: {msg1.data}")
            response1 = client.send_message(msg1, verify=args.verify)
            print(f"Response: {response1}")
            assert response1.get("received") == True, "Message should be received"
            if args.verify:
                if response1.get("checksum_valid") is not None:
                    if response1.get("checksum_valid"):
                        print("✓ String message validated and checksum verified")
                    else:
                        print(f"✗ Checksum verification failed: {response1.get('verification_error')}")
                else:
                    print("⚠ Server does not support verification (enable with --enable-verification)")
            else:
                print("✓ String message validated")

            # Example 2: Send an Int32 message
            print("\n--- Example 2: Int32 message ---")
            msg2 = Int32()
            msg2.data = 42
            print(f"Sending: {msg2.data}")
            response2 = client.send_message(msg2, verify=args.verify)
            print(f"Response: {response2}")
            assert response2.get("received") == True, "Message should be received"
            if args.verify and response2.get("checksum_valid") is not None:
                print("✓ Int32 message validated" + (" and checksum verified" if response2.get(
                    "checksum_valid") else " but checksum verification failed"))
            else:
                print("✓ Int32 message validated")

            # Example 3: Send a Point message
            print("\n--- Example 3: Point message ---")
            msg3 = Point()
            msg3.x = 1.5
            msg3.y = 2.5
            msg3.z = 3.5
            print(f"Sending: Point({msg3.x}, {msg3.y}, {msg3.z})")
            response3 = client.send_message(msg3, verify=args.verify)
            print(f"Response: {response3}")
            assert response3.get("received") == True, "Message should be received"
            if args.verify and response3.get("checksum_valid") is not None:
                print("✓ Point message validated" + (" and checksum verified" if response3.get(
                    "checksum_valid") else " but checksum verification failed"))
            else:
                print("✓ Point message validated")

            # Example 4: Send with explicit message type
            print("\n--- Example 4: Explicit message type ---")
            msg4 = String()
            msg4.data = "Explicit type test"
            response4 = client.send_message(msg4, message_type="std_msgs/String", verify=args.verify)
            print(f"Response: {response4}")
            assert response4.get("received") == True, "Message should be received"
            if args.verify and response4.get("checksum_valid") is not None:
                print("✓ Explicit type message validated" + (" and checksum verified" if response4.get(
                    "checksum_valid") else " but checksum verification failed"))
            else:
                print("✓ Explicit type message validated")

            print("\n✓ All tests passed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
