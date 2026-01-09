# LM Inference RPC

A Python package providing client-server communication utilities for VLM/LLM inference using PyTriton.

## Installation

```bash
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
```

## Structure

- `client/` - Client-side utilities for communicating with LLM/VLM servers
  - `BaseClient` - Base class for client implementations
  - `TritonClient` - Triton client using PyTriton (override `prepare_inputs()` and `process_outputs()`)
- `server/` - Server-side utilities for handling LLM/VLM inference requests
  - `BaseServer` - Base class for server implementations
  - `TritonServer` - Triton server using PyTriton (override `get_input_schema()`, `get_output_schema()`, and `inference_function()`)
- `utils/` - Shared utilities and helpers
- `examples/` - Example implementations

## Usage

### Client

```python
from lm_inference_rpc.client import TritonClient
import numpy as np

class MyClient(TritonClient):
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

### Server

```python
from lm_inference_rpc.server import TritonServer
from pytriton.model_config import Tensor
import numpy as np

class MyServer(TritonServer):
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
    server.serve()  # Blocks and serves requests
```

See `examples/` directory for complete examples.

## License

MIT

