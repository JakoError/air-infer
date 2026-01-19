# Air-Infer

A Python package providing client-server communication utilities for VLM/LLM inference and ROS2 message transmission using PyTriton.

## 🚀 Features

- **VLM/LLM Support**: High-level utilities for Vision-Language Model and Large Language Model inference
- **ROS2 Integration**: Native ROS2 message serialization/deserialization using `rclpy`
- **Triton Inference Server**: Built on PyTriton for efficient gRPC/HTTP communication
- **Flexible Architecture**: Easy to extend and customize for your specific use cases

## 📦 Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### ROS2 Requirements

For ROS2 message support, ensure you have ROS2 installed and sourced:

```bash
# Source your ROS2 installation (example for ROS2 Humble)
source /opt/ros/humble/setup.bash
```

## 📁 Structure

```
air_infer/
├── client/           # Client-side utilities
│   ├── BaseClient         # Base class for client implementations
│   ├── VLMTritonClient    # VLM/LLM client using PyTriton
│   └── ROSTritonSender    # ROS2 message sender using PyTriton
├── server/           # Server-side utilities
│   ├── BaseServer         # Base class for server implementations
│   ├── VLMTritonServer    # VLM/LLM server using PyTriton
│   └── ROSTritonReceiver  # ROS2 message receiver using PyTriton
└── utils/            # Shared utilities
    ├── vlm_utils.py       # VLM/LLM encoding/decoding utilities
    └── ros_utils.py       # ROS2 serialization utilities
```

## 📚 Usage

### ROS2 Message Transmission

#### Client (Sender)

The `ROSTritonSender` serializes ROS2 messages and sends them to the server.

```python
from air_infer.client import ROSTritonSender
from std_msgs.msg import String

# Create client
client = ROSTritonSender(
    model_name="ROSMessageHandler",
    host="127.0.0.1",
    grpc_port=9100,
)

# Send a ROS2 message
with client:
    msg = String()
    msg.data = "Hello, ROS2!"
    response = client.send_message(msg)
    print(response)  # {"received": True}
```

#### Server (Receiver)

The `ROSTritonReceiver` deserializes ROS2 messages and calls your handler function.

```python
from air_infer.server import ROSTritonReceiver
from std_msgs.msg import String

def message_handler(message):
    """Process incoming ROS2 message."""
    if isinstance(message, String):
        print(f"Received: {message.data}")
    return True  # Return True to indicate success

# Create server
server = ROSTritonReceiver(
    model_name="ROSMessageHandler",
    inference_func=message_handler,
    host="127.0.0.1",
    grpc_port=9100,
)

# Start server (blocks)
with server:
    server.start()
```

**Features:**
- Automatic message type detection and serialization
- Dynamic message type loading using `rosidl_runtime_py`
- Support for any ROS2 message type
- Type-safe message handling

### VLM/LLM Inference

#### Client

```python
from air_infer.client import VLMTritonClient
from PIL import Image

class MyClient(VLMTritonClient):
    def prepare_inputs(self, image=None, **kwargs):
        # Define your input preparation
        return {"IMAGE": np.asarray(image, dtype=np.uint8)}
    
    def process_outputs(self, outputs):
        # Define your output processing
        return {"text": outputs["TEXT_OUT"].decode()}

client = MyClient(model_name="MyModel", host="127.0.0.1", grpc_port=9100)
with client:
    result = client.infer(image=my_image)
```

#### Server

```python
from air_infer.server import VLMTritonServer
from pytriton.model_config import Tensor
import numpy as np

class MyServer(VLMTritonServer):
    def get_input_schema(self):
        return [Tensor(name="IMAGE", dtype=np.uint8, shape=(224, 224, 3))]
    
    def get_output_schema(self):
        return [Tensor(name="TEXT_OUT", dtype=bytes, shape=(1,))]
    
    def inference_function(self, IMAGE=None, **inputs):
        # Your inference logic here
        batch_size = IMAGE.shape[0]
        return {"TEXT_OUT": np.full((batch_size, 1), b"output", dtype=np.object_)}

server = MyServer(model_name="MyModel", host="127.0.0.1", grpc_port=9100)
with server:
    server.start()  # Blocks and serves requests
```

## 🧪 Examples

Complete working examples are available in the `examples/` directory:

- **`example_ros_sender.py`**: Example ROS2 client that sends various message types
- **`example_ros_receiver.py`**: Example ROS2 server that receives and processes messages
- **`example_ros_sender_perf.py`**: Performance testing tool for ROS2 message transmission
- **`example_vlm_client.py`**: Example VLM/LLM client implementation
- **`example_vlm_server.py`**: Example VLM/LLM server implementation

### Running Examples

#### ROS2 Message Transmission

**1. ROS2 Receiver (Server)**

Start the server that will receive and process ROS2 messages:

```bash
# Basic usage
python examples/example_ros_receiver.py

# With custom host/port
python examples/example_ros_receiver.py --host 0.0.0.0 --port 9100

# Silent mode (useful for performance testing)
python examples/example_ros_receiver.py --silent

# Enable message checksum verification
python examples/example_ros_receiver.py --enable-verification

# All options
python examples/example_ros_receiver.py --host 127.0.0.1 --port 9100 --model-name ROSMessageHandler --enable-verification
```

**Usage:** This server listens for incoming ROS2 messages and processes them using a message handler function. It demonstrates receiving and validating different ROS2 message types (`String`, `Int32`, `Point`). Use `--silent` to disable message printing for performance testing.

**2. ROS2 Sender (Client)**

Send ROS2 messages to the server:

```bash
# Basic usage
python examples/example_ros_sender.py

# With message verification (requires server with --enable-verification)
python examples/example_ros_sender.py --verify
```

**Usage:** This client sends multiple types of ROS2 messages (`String`, `Int32`, `Point`) and validates responses. Use `--verify` to enable checksum verification (requires the server to be started with `--enable-verification`).

**3. ROS2 Performance Testing**

Run performance tests with configurable message sizes and counts:

```bash
# Test with 100 messages of 1KB each (default)
python examples/example_ros_sender_perf.py

# Test with 1000 messages of 10KB each
python examples/example_ros_sender_perf.py -n 1000 --message-size-kb 10

# Test with 50 messages of 5MB each
python examples/example_ros_sender_perf.py -n 50 --message-size-mb 5

# Test with warmup messages and save results
python examples/example_ros_sender_perf.py -n 100 -s 1048576 -w 10 -o results.json

# Verbose output with verification
python examples/example_ros_sender_perf.py -n 100 -s 10240 -v --verify
```

**Usage:** Performance testing tool that measures latency, throughput, and bandwidth for ROS2 message transmission. Generates detailed statistics including min/max/mean/median latency, messages per second, and MB/sec bandwidth. Results are saved to a JSON file for analysis.

**Options:**
- `-n, --num-messages`: Number of messages to send (default: 100)
- `-s, --message-size`: Message size in bytes
- `--message-size-kb`: Message size in KB (alternative to `-s`)
- `--message-size-mb`: Message size in MB (alternative to `-s`)
- `-w, --warmup`: Number of warmup messages before test (default: 0)
- `--host`: Server host address (default: 127.0.0.1)
- `-p, --port`: Server gRPC port (default: 9100)
- `-m, --model-name`: Model name (default: ROSMessageHandler)
- `-o, --output`: Output file path for results (JSON format)
- `-v, --verbose`: Verbose output (print details for each message)
- `--verify`: Enable message checksum verification

#### VLM/LLM Inference

**1. VLM Server**

Start the VLM/LLM inference server:

```bash
python examples/example_vlm_server.py
```

**Usage:** This server implements a simple VLM inference function that processes media (images, videos, URLs) and JSON arguments. It demonstrates how to handle mixed media types and extract metadata. Customize the `my_inference_function` to implement your own VLM/LLM inference logic.

**2. VLM Client**

Send inference requests to the VLM server:

```bash
python examples/example_vlm_client.py
```

**Usage:** This client demonstrates sending various media types to the VLM server:
- Single PIL Image
- Multiple images and URLs
- Video (sequence of PIL Images)
- Mixed media types

The client automatically encodes media and packs additional arguments into JSON for transmission.

### Example Workflow

**ROS2 Example Workflow:**

1. Terminal 1: Start the receiver server
   ```bash
   python examples/example_ros_receiver.py --enable-verification
   ```

2. Terminal 2: Run the sender client
   ```bash
   python examples/example_ros_sender.py --verify
   ```

3. Terminal 3 (optional): Run performance tests
   ```bash
   python examples/example_ros_sender_perf.py -n 1000 --message-size-kb 10 --verify
   ```

**VLM Example Workflow:**

1. Terminal 1: Start the VLM server
   ```bash
   python examples/example_vlm_server.py
   ```

2. Terminal 2: Run the VLM client
   ```bash
   python examples/example_vlm_client.py
   ```

## 🔧 Architecture

### ROS2 Message Flow

1. **Client Side:**
   - Create ROS2 message object
   - `ROSTritonSender.send_message()` serializes message to bytes
   - Message type is auto-detected or explicitly provided
   - Serialized bytes and type are sent as tensors

2. **Server Side:**
   - `ROSTritonReceiver` receives serialized bytes and message type
   - Message type is dynamically loaded using `rosidl_runtime_py`
   - Message is deserialized to original ROS2 object
   - User-defined handler processes the message

### VLM/LLM Message Flow

1. **Encoding (Client):** High-level Python objects → Tensors
2. **Transmission:** Tensors sent via Triton Inference Server
3. **Processing (Server):** Tensors → High-level objects → Inference
4. **Encoding (Server):** Results → Tensors
5. **Decoding (Client):** Tensors → High-level results

## 📖 API Reference

### `ROSTritonSender`

**Methods:**
- `send_message(message, message_type=None)`: Send a ROS2 message
- `prepare_inputs(message, message_type=None)`: Serialize message (internal)
- `process_outputs(outputs)`: Process server response (internal)

**Parameters:**
- `model_name`: Name of the model on the server
- `host`: Server host address (default: "127.0.0.1")
- `grpc_port`: gRPC port number (default: 9100)
- `protocol`: Communication protocol ("grpc" or "http")

### `ROSTritonReceiver`

**Methods:**
- `start()`: Start the server (blocks)
- `stop()`: Stop the server

**Parameters:**
- `model_name`: Name of the model to serve
- `inference_func`: Function that processes ROS2 messages
- `host`: Server host address (default: "127.0.0.1")
- `grpc_port`: gRPC port number (default: 9100)

### ROS Utilities (`ros_utils.py`)

- `serialize_ros_message(msg)`: Serialize ROS2 message to bytes
- `deserialize_ros_message(msg_type_str, data)`: Deserialize bytes to ROS2 message
- `get_ros_message_type(msg)`: Get message type string from message object

## 🔍 Troubleshooting

### ROS2 Not Available

If you see `ImportError: ROS2 is required`, ensure:
1. ROS2 is installed
2. ROS2 environment is sourced (`source /opt/ros/humble/setup.bash`)
3. Python can find ROS2 packages

### Message Type Not Found

If deserialization fails:
- Ensure the message type string is correct (e.g., `"std_msgs/String"`)
- Verify the ROS2 package containing the message is installed
- Check that the message type matches between client and server

### Connection Issues

- Verify server is running before starting client
- Check firewall settings for gRPC port (default: 9100)
- Ensure host and port match between client and server

## 📄 License

MIT

## 👥 Contributing

Zhexian(Jako) Zhou, Yaoyu Hu

[**@AirLab CMU**](https://theairlab.org/)