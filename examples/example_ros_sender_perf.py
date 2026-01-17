"""
Performance testing example for ROS2 client using ROSTritonSender.

This example demonstrates large data transmission and performance testing
with configurable parameters for throughput, latency, and bandwidth measurements.
"""
import sys
import time
import json
import argparse
import statistics
import os
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from std_msgs.msg import String

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 messages not available. Install ROS2 or source the workspace.")
    sys.exit(1)

from lm_inference_rpc.client import ROSTritonSender


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
        if not self.latencies:
            return {
                "total_messages": self.success_count + self.failure_count,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": 0.0,
                "total_time": (self.end_time - self.start_time) if self.start_time and self.end_time else 0.0,
            }

        total_time = (self.end_time - self.start_time) if self.start_time and self.end_time else 0.0
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
            total_verified = self.verification_valid_count + self.verification_invalid_count + self.verification_missing_count
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
        results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": config,
            "statistics": self.get_stats(),
            "raw_latencies_ms": [l * 1000 for l in self.latencies],
            "raw_message_sizes_bytes": self.message_sizes,
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
    
    Efficiently generates random bytes using os.urandom() and encodes them
    as base64 to ensure valid UTF-8 string content. Base64 encoding is used
    because it's efficient and produces valid UTF-8 strings.
    
    Args:
        size_bytes: Target size in bytes for the message data
    
    Returns:
        ROS2 String message with random data of approximately the specified size
    """
    # Base64 encoding increases size by ~33% (4/3 ratio, ignoring padding)
    # Formula: base64_size ≈ (raw_bytes * 4) / 3
    # To get approximately size_bytes output: raw_bytes ≈ (size_bytes * 3) / 4
    # We add a small buffer to ensure we have enough, then truncate
    input_bytes = int(size_bytes * 3 / 4) + 1

    # Generate random bytes efficiently using os.urandom() 
    # This uses the OS's cryptographically secure random number generator
    # which is much faster than Python's random module for large data
    random_bytes = os.urandom(input_bytes)

    # Encode as base64 (produces valid UTF-8 string)
    # Base64 uses only ASCII characters (A-Z, a-z, 0-9, +, /, =) which are valid UTF-8
    data_b64 = base64.b64encode(random_bytes).decode('utf-8')

    # Truncate to target size if needed (base64 output is usually slightly larger)
    # Since each base64 char is 1 byte in UTF-8, we can truncate directly
    target_chars = size_bytes
    if len(data_b64.encode('utf-8')) > size_bytes:
        data = data_b64[:target_chars]
    else:
        data = data_b64

    msg = String()
    msg.data = data
    return msg


def run_performance_test(
        client: ROSTritonSender,
        num_messages: int,
        message_size_bytes: int,
        warmup_messages: int = 0,
        verbose: bool = False,
        verify: bool = False,
) -> PerformanceRecorder:
    """
    Run performance test with specified parameters.
    
    Args:
        client: ROSTritonSender client instance
        num_messages: Number of messages to send
        message_size_bytes: Size of each message in bytes
        warmup_messages: Number of warmup messages to send before test
        verbose: If True, print details for each message
        verify: If True, enable checksum verification for each message
    
    Returns:
        PerformanceRecorder with test results
    """
    recorder = PerformanceRecorder()

    # Warmup phase
    if warmup_messages > 0:
        print(f"Sending {warmup_messages} warmup messages...")
        for i in range(warmup_messages):
            try:
                msg = generate_large_string_message(message_size_bytes)
                client.send_message(msg, verify=verify)
                if verbose and (i + 1) % max(1, warmup_messages // 10) == 0:
                    print(f"  Warmup {i + 1}/{warmup_messages}")
            except Exception as e:
                print(f"  Warmup message {i + 1} failed: {e}")

    # Actual test phase
    print(f"\nStarting performance test:")
    print(f"  Messages: {num_messages}")
    print(f"  Message size: {message_size_bytes:,} bytes ({message_size_bytes / 1024:.2f} KB)")
    print(f"  Total data: {num_messages * message_size_bytes / (1024 * 1024):.2f} MB")
    if verify:
        print(f"  Verification: ENABLED")
    print()

    recorder.start_test()

    for i in range(num_messages):
        msg = generate_large_string_message(message_size_bytes)
        msg_start_time = time.time()

        try:
            response = client.send_message(msg, verify=verify)
            msg_end_time = time.time()
            latency = msg_end_time - msg_start_time

            # Get actual serialized message size
            from lm_inference_rpc.utils.ros_utils import serialize_ros_message
            actual_size = len(serialize_ros_message(msg))

            # Extract verification result
            checksum_valid = None
            if verify:
                checksum_valid = response.get("checksum_valid")
                if checksum_valid is None:
                    # Server doesn't support verification or didn't return checksum
                    checksum_valid = None
                    if verbose:
                        print(f"  Warning: Verification enabled but server response missing checksum")

            recorder.record_message(latency, actual_size, success=True, checksum_valid=checksum_valid)

            if verbose:
                verify_status = ""
                if verify:
                    if checksum_valid is True:
                        verify_status = ", ✓ Verified"
                    elif checksum_valid is False:
                        verify_status = f", ✗ Verification failed: {response.get('verification_error', 'checksum mismatch')}"
                    else:
                        verify_status = ", ⚠ Verification not available"
                print(f"Message {i + 1}/{num_messages}: "
                      f"Latency={latency * 1000:.2f}ms, "
                      f"Size={actual_size:,} bytes{verify_status}")
            elif (i + 1) % max(1, num_messages // 10) == 0:
                verify_info = ""
                if verify:
                    verify_info = f", Verified: {recorder.verification_valid_count}/{recorder.verification_valid_count + recorder.verification_invalid_count}"
                print(f"Progress: {i + 1}/{num_messages} messages sent "
                      f"({(i + 1) / num_messages * 100:.1f}%){verify_info}")

        except Exception as e:
            msg_end_time = time.time()
            latency = msg_end_time - msg_start_time
            recorder.record_message(latency, message_size_bytes, success=False, checksum_valid=None)
            print(f"Message {i + 1} failed: {e}")

    recorder.end_test()
    return recorder


def print_statistics(stats: Dict[str, Any]):
    """Print formatted performance statistics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 60)

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
        print(f"  Total Data:     {bw['total_mb']:.2f} MB ({bw['total_bytes']:,} bytes)")
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

    print("=" * 60 + "\n")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Performance testing tool for ROS2 Triton client with large data transmission",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 100 messages of 1MB each
  python example_ros_sender_perf.py -n 100 -s 1048576
  
  # Test with 1000 messages of 10KB each, save to results.json
  python example_ros_sender_perf.py -n 1000 -s 10240 -o results.json
  
  # Test with 50 messages of 5MB each, with warmup and verbose output
  python example_ros_sender_perf.py -n 50 -s 5242880 -w 5 -v
        """
    )

    parser.add_argument(
        '-n', '--num-messages',
        type=int,
        default=100,
        help='Number of messages to send (default: 100)'
    )

    parser.add_argument(
        '-s', '--message-size',
        type=int,
        default=1024,
        help='Size of each message in bytes (default: 1024)'
    )

    parser.add_argument(
        '--message-size-kb',
        type=float,
        default=None,
        help='Size of each message in KB (alternative to --message-size)'
    )

    parser.add_argument(
        '--message-size-mb',
        type=float,
        default=None,
        help='Size of each message in MB (alternative to --message-size)'
    )

    parser.add_argument(
        '-w', '--warmup',
        type=int,
        default=0,
        help='Number of warmup messages to send before test (default: 0)'
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

    # Calculate message size
    message_size_bytes = args.message_size
    if args.message_size_kb is not None:
        message_size_bytes = int(args.message_size_kb * 1024)
    elif args.message_size_mb is not None:
        message_size_bytes = int(args.message_size_mb * 1024 * 1024)

    # Generate output filename if not provided
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{timestamp}.json"

    # Configuration dictionary
    config = {
        "num_messages": args.num_messages,
        "message_size_bytes": message_size_bytes,
        "warmup_messages": args.warmup,
        "host": args.host,
        "port": args.port,
        "model_name": args.model_name,
        "verbose": args.verbose,
        "verify": args.verify,
    }

    print("Performance Test Configuration:")
    print(f"  Model:          {config['model_name']}")
    print(f"  Host:           {config['host']}")
    print(f"  Port:           {config['port']}")
    print(f"  Messages:       {config['num_messages']}")
    print(f"  Message Size:   {message_size_bytes:,} bytes ({message_size_bytes / 1024:.2f} KB)")
    print(f"  Warmup:         {config['warmup_messages']}")
    print(f"  Verification:   {'ENABLED' if args.verify else 'DISABLED'}")
    print(f"  Output File:    {output_file}")
    if args.verify:
        print(f"  Note: Server must be run with --enable-verification for verification to work")
    print()

    # Create client
    print(f"Connecting to {config['host']}:{config['port']}...")
    client = ROSTritonSender(
        model_name=config['model_name'],
        host=config['host'],
        grpc_port=config['port'],
    )

    try:
        with client:
            print("Connected! Starting test...\n")

            # Run performance test
            recorder = run_performance_test(
                client=client,
                num_messages=config['num_messages'],
                message_size_bytes=message_size_bytes,
                warmup_messages=config['warmup_messages'],
                verbose=config['verbose'],
                verify=config['verify'],
            )

            # Get statistics
            stats = recorder.get_stats()

            # Print results
            print_statistics(stats)

            # Save results
            recorder.save_to_file(output_file, config)

            print("\nTest completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
