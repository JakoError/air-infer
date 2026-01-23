"""
Performance testing example for ROS2 client using ROSTritonSender with multiple models.

This example demonstrates performance testing with:
- Multiple models on the same or different Triton instances
- Parallel senders per model
- Support for ROS Image messages with configurable size
"""
import sys
import time
import json
import argparse
import statistics
import os
import base64
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

try:
    from std_msgs.msg import String
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 messages not available. Install ROS2 or source the workspace.")
    sys.exit(1)

from air_infer.client import ROSTritonSender


class PerformanceRecorder:
    """Record and analyze performance metrics."""

    def __init__(self):
        self.latencies: List[float] = []
        self.message_sizes: List[int] = []
        self.success_count = 0
        self.failure_count = 0
        self.verification_enabled = False
        self.verification_valid_count = 0
        self.verification_invalid_count = 0
        self.verification_missing_count = 0
        self.start_time: float = None
        self.end_time: float = None
        self.lock = threading.Lock()

    def start_test(self):
        """Mark the start of the test."""
        self.start_time = time.time()

    def end_test(self):
        """Mark the end of the test."""
        self.end_time = time.time()

    def record_message(
            self,
            latency: float,
            message_size: int,
            success: bool = True,
            checksum_valid: Optional[bool] = None
    ):
        """
        Record metrics for a single message.

        Args:
            latency: Message latency in seconds
            message_size: Message size in bytes
            success: Whether message was successfully sent/received
            checksum_valid: Whether checksum verification passed (None if verification not enabled)
        """
        with self.lock:
            if success:
                self.latencies.append(latency)
                self.message_sizes.append(message_size)
                self.success_count += 1

                # Track verification if enabled
                if checksum_valid is not None:
                    self.verification_enabled = True
                    if checksum_valid:
                        self.verification_valid_count += 1
                    else:
                        self.verification_invalid_count += 1
                elif self.verification_enabled and checksum_valid is None:
                    # Verification was enabled but not returned in response
                    self.verification_missing_count += 1
            else:
                self.failure_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Calculate and return performance statistics."""
        with self.lock:
            if not self.latencies:
                return {
                    "total_messages": self.success_count + self.failure_count,
                    "success_count": self.success_count,
                    "failure_count": self.failure_count,
                    "success_rate": 0.0,
                    "total_time": (self.end_time - self.start_time) if self.start_time and self.end_time else 0.0,
                }

            total_time = (
                self.end_time - self.start_time) if self.start_time and self.end_time else 0.0
            total_bytes = sum(self.message_sizes)
            total_messages = self.success_count + self.failure_count

            stats = {
                "total_messages": total_messages,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": self.success_count / total_messages if total_messages > 0 else 0.0,
                "total_time_seconds": total_time,
                "latency_ms": {
                    "min": min(self.latencies) * 1000,
                    "max": max(self.latencies) * 1000,
                    "mean": statistics.mean(self.latencies) * 1000,
                    "median": statistics.median(self.latencies) * 1000,
                    "p50": statistics.median(self.latencies) * 1000,
                    "p95": self._percentile(self.latencies, 0.95) * 1000,
                    "p99": self._percentile(self.latencies, 0.99) * 1000,
                },
                "throughput": {
                    "messages_per_second": self.success_count / total_time if total_time > 0 else 0.0,
                },
                "bandwidth": {
                    "bytes_per_second": total_bytes / total_time if total_time > 0 else 0.0,
                    "kb_per_second": (total_bytes / total_time / 1024) if total_time > 0 else 0.0,
                    "mb_per_second": (total_bytes / total_time / (1024 * 1024)) if total_time > 0 else 0.0,
                    "total_bytes": total_bytes,
                    "total_kb": total_bytes / 1024,
                    "total_mb": total_bytes / (1024 * 1024),
                },
                "message_size": {
                    "mean_bytes": statistics.mean(self.message_sizes),
                    "min_bytes": min(self.message_sizes),
                    "max_bytes": max(self.message_sizes),
                }
            }

            # Add verification statistics if enabled
            if self.verification_enabled:
                total_verified = self.verification_valid_count + \
                    self.verification_invalid_count + self.verification_missing_count
                stats["verification"] = {
                    "enabled": True,
                    "valid_count": self.verification_valid_count,
                    "invalid_count": self.verification_invalid_count,
                    "missing_count": self.verification_missing_count,
                    "total_verified": total_verified,
                    "validity_rate": self.verification_valid_count / total_verified if total_verified > 0 else 0.0,
                }
            else:
                stats["verification"] = {
                    "enabled": False,
                }

            return stats

    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]

    def save_to_file(self, filepath: str, config: Dict[str, Any]):
        """Save results to a JSON file."""
        # Get statistics first (handles its own locking)
        stats = self.get_stats()
        
        # Get raw data with proper locking
        with self.lock:
            raw_latencies_ms = [l * 1000 for l in self.latencies]
            raw_message_sizes_bytes = self.message_sizes.copy()
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": config,
            "statistics": stats,
            "raw_latencies_ms": raw_latencies_ms,
            "raw_message_sizes_bytes": raw_message_sizes_bytes,
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path


def generate_large_string_message(size_bytes: int) -> String:
    """
    Generate a ROS2 String message with random data of specified size.

    Args:
        size_bytes: Target size in bytes for the message data

    Returns:
        ROS2 String message with random data of approximately the specified size
    """
    input_bytes = int(size_bytes * 3 / 4) + 1
    random_bytes = os.urandom(input_bytes)
    data_b64 = base64.b64encode(random_bytes).decode('utf-8')

    target_chars = size_bytes
    if len(data_b64.encode('utf-8')) > size_bytes:
        data = data_b64[:target_chars]
    else:
        data = data_b64

    msg = String()
    msg.data = data
    return msg


def generate_image_message(width: int, height: int, encoding: str = "rgb8") -> Image:
    """
    Generate a ROS2 Image message with specified dimensions.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        encoding: Pixel encoding format (default: "rgb8")

    Returns:
        ROS2 Image message with random pixel data
    """
    msg = Image()
    msg.header = Header()
    msg.header.stamp.sec = int(time.time())
    msg.header.stamp.nanosec = int((time.time() % 1) * 1e9)
    msg.header.frame_id = "test_camera"

    msg.width = width
    msg.height = height
    msg.encoding = encoding
    msg.is_bigendian = 0

    # Calculate step (bytes per row)
    if encoding == "rgb8" or encoding == "bgr8":
        bytes_per_pixel = 3
    elif encoding == "rgba8" or encoding == "bgra8":
        bytes_per_pixel = 4
    elif encoding == "mono8":
        bytes_per_pixel = 1
    else:
        # Default to 3 bytes per pixel
        bytes_per_pixel = 3

    msg.step = width * bytes_per_pixel

    # Generate random pixel data
    total_bytes = height * msg.step
    msg.data = os.urandom(total_bytes)

    return msg


def run_performance_test_for_model(
        client: ROSTritonSender,
        model_name: str,
        num_messages: int,
        message_size_bytes: int,
        use_image: bool,
        image_width: int,
        image_height: int,
        recorder: PerformanceRecorder,
        verbose: bool = False,
        verify: bool = False,
):
    """
    Run performance test for a single model.

    Args:
        client: ROSTritonSender client instance
        model_name: Name of the model
        num_messages: Number of messages to send
        message_size_bytes: Size of each message in bytes (for String messages)
        use_image: If True, send Image messages instead of String
        image_width: Image width (if use_image is True)
        image_height: Image height (if use_image is True)
        recorder: PerformanceRecorder instance
        verbose: If True, print details for each message
        verify: If True, enable checksum verification
    """
    from air_infer.utils.ros_utils import serialize_ros_message

    for i in range(num_messages):
        if use_image:
            msg = generate_image_message(image_width, image_height)
        else:
            msg = generate_large_string_message(message_size_bytes)

        msg_start_time = time.time()

        try:
            response = client.send_message(msg, verify=verify)
            msg_end_time = time.time()
            latency = msg_end_time - msg_start_time

            # Get actual serialized message size
            actual_size = len(serialize_ros_message(msg))

            # Extract verification result
            checksum_valid = None
            if verify:
                checksum_valid = response.get("checksum_valid")

            recorder.record_message(
                latency, actual_size, success=True, checksum_valid=checksum_valid)

            if verbose and (i + 1) % max(1, num_messages // 10) == 0:
                verify_status = ""
                if verify:
                    if checksum_valid is True:
                        verify_status = ", ✓ Verified"
                    elif checksum_valid is False:
                        verify_status = ", ✗ Verification failed"
                    else:
                        verify_status = ", ⚠ Verification not available"
                msg_type = "Image" if use_image else "String"
                print(f"[{model_name}] Message {i + 1}/{num_messages} ({msg_type}): "
                      f"Latency={latency * 1000:.2f}ms, Size={actual_size:,} bytes{verify_status}")

        except Exception as e:
            msg_end_time = time.time()
            latency = msg_end_time - msg_start_time
            recorder.record_message(
                latency, message_size_bytes, success=False, checksum_valid=None)
            if verbose:
                print(f"[{model_name}] Message {i + 1} failed: {e}")


def print_statistics(stats: Dict[str, Any]):
    """Print formatted performance statistics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 70)

    print(f"\nMessages:")
    print(f"  Total:      {stats['total_messages']}")
    print(f"  Successful: {stats['success_count']}")
    print(f"  Failed:     {stats['failure_count']}")
    print(f"  Success Rate: {stats['success_rate'] * 100:.2f}%")

    print(f"\nTiming:")
    print(f"  Total Time: {stats['total_time_seconds']:.3f} seconds")

    if 'latency_ms' in stats:
        lat = stats['latency_ms']
        print(f"\nLatency (ms):")
        print(f"  Min:   {lat['min']:.2f}")
        print(f"  Max:   {lat['max']:.2f}")
        print(f"  Mean:  {lat['mean']:.2f}")
        print(f"  Median: {lat['median']:.2f}")
        print(f"  P95:   {lat['p95']:.2f}")
        print(f"  P99:   {lat['p99']:.2f}")

    if 'throughput' in stats:
        tp = stats['throughput']
        print(f"\nThroughput:")
        print(f"  Messages/sec: {tp['messages_per_second']:.2f}")

    if 'bandwidth' in stats:
        bw = stats['bandwidth']
        print(f"\nBandwidth:")
        print(
            f"  Total Data:     {bw['total_mb']:.2f} MB ({bw['total_bytes']:,} bytes)")
        print(f"  Bytes/sec:      {bw['bytes_per_second']:,.0f}")
        print(f"  KB/sec:         {bw['kb_per_second']:.2f}")
        print(f"  MB/sec:         {bw['mb_per_second']:.2f}")

    if 'message_size' in stats:
        ms = stats['message_size']
        print(f"\nMessage Size:")
        print(f"  Mean: {ms['mean_bytes']:,.0f} bytes")
        print(f"  Min:  {ms['min_bytes']:,.0f} bytes")
        print(f"  Max:  {ms['max_bytes']:,.0f} bytes")

    if 'verification' in stats and stats['verification'].get('enabled'):
        vf = stats['verification']
        print(f"\nVerification:")
        print(f"  Valid:     {vf['valid_count']}")
        print(f"  Invalid:   {vf['invalid_count']}")
        print(f"  Missing:   {vf['missing_count']}")
        print(f"  Validity Rate: {vf['validity_rate'] * 100:.2f}%")

    print("=" * 70 + "\n")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Performance testing tool for ROS2 Triton client with multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 3 models on 1 instance, 10 messages each, 2 parallel senders per model
  python example_ros_sender_perf_multi.py --num-models 3 --num-instances 1 -n 10 --parallel-senders 2
  
  # Test with Image messages (640x480)
  python example_ros_sender_perf_multi.py --num-models 2 --use-image --image-width 640 --image-height 480 -n 100
  
  # Test 5 models across 2 instances
  python example_ros_sender_perf_multi.py --num-models 5 --num-instances 2 -n 50
        """
    )

    parser.add_argument(
        '-n', '--num-messages',
        type=int,
        default=100,
        help='Number of messages to send per sender (default: 100)'
    )

    parser.add_argument(
        '-s', '--message-size',
        type=int,
        default=1024,
        help='Size of each String message in bytes (default: 1024, ignored if --use-image)'
    )

    parser.add_argument(
        '--num-models',
        type=int,
        default=1,
        help='Number of models to test (default: 1)'
    )

    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='Number of Triton instances (default: 1). Models will be distributed across instances'
    )

    parser.add_argument(
        '--parallel-senders',
        type=int,
        default=1,
        help='Number of parallel senders per model (default: 1)'
    )

    parser.add_argument(
        '--model-prefix',
        type=str,
        default='Model',
        help='Prefix for model names (default: Model). Models will be named Model0, Model1, etc.'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Server host address (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--base-port',
        type=int,
        default=9100,
        help='Base port number (default: 9100). Each instance uses 3 consecutive ports: '
             'Instance 0: base_port, base_port+1, base_port+2; '
             'Instance 1: base_port+3, base_port+4, base_port+5; etc.'
    )

    parser.add_argument(
        '--use-image',
        action='store_true',
        help='Use ROS Image messages instead of String messages'
    )

    parser.add_argument(
        '--image-width',
        type=int,
        default=640,
        help='Image width in pixels (default: 640, only used with --use-image)'
    )

    parser.add_argument(
        '--image-height',
        type=int,
        default=480,
        help='Image height in pixels (default: 480, only used with --use-image)'
    )

    parser.add_argument(
        '-w', '--warmup',
        type=int,
        default=0,
        help='Number of warmup messages to send before test (default: 0)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path for results (JSON format). Default: results_<timestamp>.json'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output (print details for each message)'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Enable message checksum verification (requires server with --enable-verification)'
    )

    args = parser.parse_args()

    if args.num_models < 1:
        print("Error: --num-models must be at least 1")
        sys.exit(1)

    if args.num_instances < 1:
        print("Error: --num-instances must be at least 1")
        sys.exit(1)

    if args.parallel_senders < 1:
        print("Error: --parallel-senders must be at least 1")
        sys.exit(1)

    # Calculate message size for Image messages
    if args.use_image:
        # Calculate approximate size: width * height * bytes_per_pixel + header
        bytes_per_pixel = 3  # rgb8
        image_size_bytes = args.image_width * args.image_height * \
            bytes_per_pixel + 100  # +100 for header
        message_size_bytes = image_size_bytes
    else:
        message_size_bytes = args.message_size

    # Generate output filename if not provided
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_multi_{timestamp}.json"

    # Configuration dictionary
    config = {
        "num_messages": args.num_messages,
        "message_size_bytes": message_size_bytes,
        "num_models": args.num_models,
        "num_instances": args.num_instances,
        "parallel_senders": args.parallel_senders,
        "model_prefix": args.model_prefix,
        "warmup_messages": args.warmup,
        "host": args.host,
        "base_port": args.base_port,
        "use_image": args.use_image,
        "image_width": args.image_width if args.use_image else None,
        "image_height": args.image_height if args.use_image else None,
        "verbose": args.verbose,
        "verify": args.verify,
    }

    print("=" * 70)
    print("Multi-Model Performance Test Configuration")
    print("=" * 70)
    print(f"  Models:            {config['num_models']}")
    print(f"  Instances:         {config['num_instances']}")
    print(f"  Parallel Senders:  {config['parallel_senders']} per model")
    print(
        f"  Total Senders:     {config['num_models'] * config['parallel_senders']}")
    print(f"  Messages/Sender:  {config['num_messages']}")
    print(
        f"  Total Messages:    {config['num_models'] * config['parallel_senders'] * config['num_messages']}")
    print(f"  Message Type:      {'Image' if args.use_image else 'String'}")
    if args.use_image:
        print(f"  Image Size:        {args.image_width}x{args.image_height}")
        print(
            f"  Approx Size:       {message_size_bytes:,} bytes ({message_size_bytes / 1024:.2f} KB)")
    else:
        print(
            f"  Message Size:      {message_size_bytes:,} bytes ({message_size_bytes / 1024:.2f} KB)")
    print(f"  Host:              {config['host']}")
    print(f"  Base Port:         {config['base_port']}")
    print(f"\n  Port Configuration:")
    for instance_id in range(args.num_instances):
        port_offset = instance_id * 3
        grpc_port = args.base_port + port_offset
        http_port = args.base_port + port_offset + 1
        metrics_port = args.base_port + port_offset + 2
        print(f"    Instance {instance_id}: gRPC={grpc_port}, HTTP={http_port}, Metrics={metrics_port}")
    print(f"  Verification:      {'ENABLED' if args.verify else 'DISABLED'}")
    print(f"  Output File:        {output_file}")
    print()

    # Calculate models per instance
    models_per_instance = args.num_models // args.num_instances
    extra_models = args.num_models % args.num_instances

    # Create shared recorder
    recorder = PerformanceRecorder()

    # Create clients and threads
    clients = []
    threads = []

    model_idx = 0
    for instance_id in range(args.num_instances):
        num_models_this_instance = models_per_instance + \
            (1 if instance_id < extra_models else 0)

        if num_models_this_instance == 0:
            continue

        # Calculate ports for this instance
        # Each instance uses 3 consecutive ports: grpc, http, metrics
        # Instance 0: base_port, base_port+1, base_port+2
        # Instance 1: base_port+3, base_port+4, base_port+5
        # Instance 2: base_port+6, base_port+7, base_port+8
        # etc.
        port_offset = instance_id * 3
        grpc_port = args.base_port + port_offset
        http_port = args.base_port + port_offset + 1
        metrics_port = args.base_port + port_offset + 2

        if args.verbose:
            print(f"Instance {instance_id}: gRPC={grpc_port}, HTTP={http_port}, Metrics={metrics_port}, Models: {num_models_this_instance}")

        for model_offset in range(num_models_this_instance):
            model_name = f"{args.model_prefix}{model_idx}"
            model_idx += 1

            # Create clients for parallel senders
            for sender_id in range(args.parallel_senders):
                client = ROSTritonSender(
                    model_name=model_name,
                    host=args.host,
                    grpc_port=grpc_port,
                )
                clients.append((client, model_name, grpc_port))

    print(
        f"Created {len(clients)} clients across {args.num_instances} instance(s)")
    print("Connecting clients...")

    # Connect all clients
    for client, model_name, port in clients:
        try:
            client.connect()
        except Exception as e:
            print(
                f"Warning: Failed to connect client for {model_name} on port {port}: {e}")

    print("All clients connected!")

    # Warmup phase
    if args.warmup > 0:
        print(f"\nSending {args.warmup} warmup messages per sender...")
        for client, model_name, _ in clients:
            for i in range(args.warmup):
                try:
                    if args.use_image:
                        msg = generate_image_message(
                            args.image_width, args.image_height)
                    else:
                        msg = generate_large_string_message(message_size_bytes)
                    client.send_message(msg, verify=args.verify)
                except Exception as e:
                    if args.verbose:
                        print(f"  Warmup failed for {model_name}: {e}")

    print(f"\nStarting performance test...")
    recorder.start_test()

    # Start all sender threads
    def run_sender(client, model_name):
        run_performance_test_for_model(
            client=client,
            model_name=model_name,
            num_messages=args.num_messages,
            message_size_bytes=message_size_bytes,
            use_image=args.use_image,
            image_width=args.image_width,
            image_height=args.image_height,
            recorder=recorder,
            verbose=args.verbose,
            verify=args.verify,
        )

    for client, model_name, _ in clients:
        thread = threading.Thread(target=run_sender, args=(
            client, model_name), daemon=True)
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    recorder.end_test()

    # Disconnect all clients
    for client, _, _ in clients:
        try:
            client.disconnect()
        except:
            pass

    # Get statistics
    stats = recorder.get_stats()

    # Print results
    print_statistics(stats)

    # Save results
    recorder.save_to_file(output_file, config)

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
