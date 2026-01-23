"""
Example ROS2 receiver server for testing multiple models on same Triton instance.

This example demonstrates how to run multiple ROS2 receivers that share
the same Triton instance, each handling a different model.
"""
import sys
import threading
import time
import argparse
import os
from typing import List

try:
    from std_msgs.msg import String, Int32
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 messages not available. Install ROS2 or source the workspace.")
    sys.exit(1)

from air_infer.server import ROSTritonReceiver

# Global counters for validation
received_messages = {}
received_locks = {}
VERBOSE_OUTPUT = True


def create_message_handler(model_name: str):
    """
    Create a message handler function for a specific model.
    
    Args:
        model_name: Name of the model this handler is for
        
    Returns:
        Handler function
    """
    if model_name not in received_messages:
        received_messages[model_name] = []
        received_locks[model_name] = threading.Lock()
    
    def message_handler(message):
        """
        Handle incoming ROS2 messages.
        
        Args:
            message: Deserialized ROS2 message object
            
        Returns:
            True if message was processed successfully
        """
        global VERBOSE_OUTPUT
        
        with received_locks[model_name]:
            msg_info = {
                "type": message.__class__.__name__,
                "data": str(message)[:100] if len(str(message)) > 100 else str(message),  # Truncate for display
                "timestamp": time.time()
            }
            received_messages[model_name].append(msg_info)
            
            if VERBOSE_OUTPUT:
                if isinstance(message, Image):
                    print(f"[{model_name}] Received Image: {message.width}x{message.height}, "
                          f"encoding={message.encoding}, size={len(message.data)} bytes")
                else:
                    print(f"[{model_name}] Received: {msg_info['type']} - {msg_info['data']}")
        
        # Validate message content based on type
        if isinstance(message, String):
            assert hasattr(message, 'data'), "String message should have 'data' field"
            assert isinstance(message.data, str), "String.data should be a string"
            if VERBOSE_OUTPUT:
                print(f"  ✓ [{model_name}] Validated String message")
        
        elif isinstance(message, Int32):
            assert hasattr(message, 'data'), "Int32 message should have 'data' field"
            assert isinstance(message.data, int), "Int32.data should be an integer"
            if VERBOSE_OUTPUT:
                print(f"  ✓ [{model_name}] Validated Int32 message: {message.data}")
        
        elif isinstance(message, Point):
            assert hasattr(message, 'x'), "Point message should have 'x' field"
            assert hasattr(message, 'y'), "Point message should have 'y' field"
            assert hasattr(message, 'z'), "Point message should have 'z' field"
            if VERBOSE_OUTPUT:
                print(f"  ✓ [{model_name}] Validated Point message: ({message.x}, {message.y}, {message.z})")
        
        elif isinstance(message, Image):
            assert hasattr(message, 'width'), "Image message should have 'width' field"
            assert hasattr(message, 'height'), "Image message should have 'height' field"
            assert hasattr(message, 'data'), "Image message should have 'data' field"
            assert len(message.data) == message.height * message.step, \
                f"Image data size mismatch: expected {message.height * message.step}, got {len(message.data)}"
            if VERBOSE_OUTPUT:
                print(f"  ✓ [{model_name}] Validated Image message: {message.width}x{message.height}")
        
        return True
    
    return message_handler


def main():
    """Example usage of multiple ROSTritonReceivers sharing Triton instances."""
    global VERBOSE_OUTPUT
    
    parser = argparse.ArgumentParser(
        description="ROS2 Triton receiver server example for multiple models",
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
        '--base-port',
        type=int,
        default=9100,
        help='Base port number (default: 9100). Each instance uses 3 consecutive ports: '
             'Instance 0: base_port, base_port+1, base_port+2; '
             'Instance 1: base_port+3, base_port+4, base_port+5; etc.'
    )
    
    parser.add_argument(
        '--num-models',
        type=int,
        default=1,
        help='Number of models to create (default: 1)'
    )
    
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='Number of Triton instances to create (default: 1). Models will be distributed across instances'
    )
    
    parser.add_argument(
        '--model-prefix',
        type=str,
        default='Model',
        help='Prefix for model names (default: Model). Models will be named Model0, Model1, etc.'
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
    
    if args.num_models < 1:
        print("Error: --num-models must be at least 1")
        sys.exit(1)
    
    if args.num_instances < 1:
        print("Error: --num-instances must be at least 1")
        sys.exit(1)
    
    if args.num_instances > args.num_models:
        print(f"Warning: --num-instances ({args.num_instances}) > --num-models ({args.num_models}). "
              f"Some instances will have no models.")
    
    print("=" * 70)
    print("Multiple Models ROS2 Triton Receiver")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Host:              {args.host}")
    print(f"  Base Port:          {args.base_port}")
    print(f"  Number of Models:   {args.num_models}")
    print(f"  Number of Instances: {args.num_instances}")
    print(f"  Model Prefix:      {args.model_prefix}")
    print(f"\n  Port Configuration:")
    for instance_id in range(args.num_instances):
        port_offset = instance_id * 3
        grpc_port = args.base_port + port_offset
        http_port = args.base_port + port_offset + 1
        metrics_port = args.base_port + port_offset + 2
        print(f"    Instance {instance_id}: gRPC={grpc_port}, HTTP={http_port}, Metrics={metrics_port}")
    print(f"  Verification:      {'ENABLED' if args.enable_verification else 'DISABLED'}")
    print(f"  Verbose Output:    {'ENABLED' if VERBOSE_OUTPUT else 'DISABLED'}")
    print()
    
    # Calculate models per instance
    models_per_instance = args.num_models // args.num_instances
    extra_models = args.num_models % args.num_instances
    
    servers: List[ROSTritonReceiver] = []
    server_threads: List[threading.Thread] = []
    
    model_idx = 0
    for instance_id in range(args.num_instances):
        # Calculate number of models for this instance
        num_models_this_instance = models_per_instance + (1 if instance_id < extra_models else 0)
        
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
        
        print(f"Instance {instance_id}: gRPC={grpc_port}, HTTP={http_port}, Metrics={metrics_port}, Models: {num_models_this_instance}")
        
        # Create servers for this instance
        for model_offset in range(num_models_this_instance):
            model_name = f"{args.model_prefix}{model_idx}"
            model_idx += 1
            
            print(f"  Creating server for model '{model_name}'...")
            
            server = ROSTritonReceiver(
                model_name=model_name,
                inference_func=create_message_handler(model_name),
                host=args.host,
                grpc_port=grpc_port,
                http_port=http_port,
                metrics_port=metrics_port,
                log_verbose=0,
                enable_verification=args.enable_verification,
            )
            
            servers.append(server)
            
            # Start server in a thread
            def start_server(srv, name):
                def run():
                    print(f"Starting server for '{name}'...")
                    srv.start()
                return run
            
            thread = threading.Thread(target=start_server(server, model_name), daemon=True)
            server_threads.append(thread)
            thread.start()
            
            # Small delay to ensure proper initialization order
            time.sleep(0.1)
    
    print()
    print("=" * 70)
    print("All servers started!")
    print("=" * 70)
    print("\nServer configuration:")
    for i, server in enumerate(servers):
        model_name = f"{args.model_prefix}{i}"
        print(f"  - {model_name}: {args.host}:{server.grpc_port} (gRPC), {server.http_port} (HTTP), {server.metrics_port} (Metrics)")
    print("\nWaiting for messages...")
    if not VERBOSE_OUTPUT:
        print("(Silent mode: received messages will not be printed)\n")
    print("Press Ctrl+C to stop...\n")
    
    try:
        # Keep running until interrupted
        last_counts = {}
        while True:
            time.sleep(1)
            
            if not VERBOSE_OUTPUT:
                # Print summary in silent mode
                total_received = 0
                for model_name in received_messages:
                    with received_locks[model_name]:
                        count = len(received_messages[model_name])
                        total_received += count
                        if count != last_counts.get(model_name, 0):
                            print(f"[{model_name}] Total messages: {count}")
                            last_counts[model_name] = count
                
                if total_received > 0:
                    print(f"Total across all models: {total_received}")
    
    except KeyboardInterrupt:
        print("\n\nStopping all servers...")
        for server in servers:
            server.stop()
        
        print("\n" + "=" * 70)
        print("Final Statistics")
        print("=" * 70)
        total_messages = 0
        for model_name in sorted(received_messages.keys()):
            with received_locks[model_name]:
                count = len(received_messages[model_name])
                total_messages += count
                print(f"  {model_name}: {count} messages")
        print(f"\nTotal messages processed: {total_messages}")
        print("All servers stopped.")


if __name__ == "__main__":
    main()
