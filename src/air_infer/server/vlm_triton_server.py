"""
Triton server implementation using PyTriton.
"""
from typing import Dict, List, Optional, Callable, Tuple, Union, Any
import json
import numpy as np
from pytriton.decorators import sample
from pytriton.model_config import Tensor, ModelConfig
from pytriton.triton import Triton, TritonConfig

from .base_server import BaseServer
from ..utils.vlm_utils import decode_request_tensors, encode_response
from ..utils.triton_manager import TritonInstanceManager


class VLMTritonServer(BaseServer):
    """
    Triton server for LLM/VLM inference using PyTriton.

    This class provides a framework for Triton-based serving.
    
    Usage options:
    1. Pass an inference function directly (recommended):
       ```python
       def my_inference_func(IMAGE=None, VIDEO=None, MEDIA_URLS=None, **inputs):
           # Your inference logic
           return {"RESULTS_JSON": ...}
       
       server = TritonServer(
           model_name="MyModel",
           inference_func=my_inference_func
       )
       ```
    
    2. Subclass and override methods:
       ```python
       class MyServer(TritonServer):
           def inference_function(self, **inputs):
               # Your inference logic
               return {"RESULTS_JSON": ...}
       ```
    
    The inference function should accept input tensors as keyword arguments
    (with batch dimension) and return a dictionary of output tensors.
    The @batch decorator is automatically applied.
    """

    def __init__(
            self,
            model_name: str,
            inference_func: Optional[Callable[..., Dict[str, np.ndarray]]] = None,
            host: str = "127.0.0.1",
            grpc_port: int = 9100,
            http_port: int = 8100,
            metrics_port: int = 8101,
            log_verbose: int = 1,
            enable_grpc: bool = True,
            enable_http: bool = True,
    ):
        """
        Initialize the Triton server.

        Args:
            model_name: Name of the model to serve
            inference_func: Optional inference function to use. If provided, this function
                will be used instead of the inference_function() method. The function should
                accept input tensors as keyword arguments (e.g., IMAGE, VIDEO, MEDIA_URLS, ARGS_JSON)
                and return a dictionary of output tensors (e.g., {"RESULTS_JSON": ...}).
                The function will automatically receive batched inputs when @batch is applied.
                If None, subclasses must override inference_function() method.
            host: Server host address
            grpc_port: gRPC port number
            http_port: HTTP port number
            metrics_port: Metrics port number
            log_verbose: Log verbosity level
            enable_grpc: Enable gRPC protocol (default: True)
            enable_http: Enable HTTP protocol (default: True)
        """
        super().__init__(host=host, port=grpc_port)
        self.model_name = model_name
        self.inference_func = inference_func
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.metrics_port = metrics_port
        self.log_verbose = log_verbose
        self.enable_grpc = enable_grpc
        self.enable_http = enable_http

        self._triton: Optional[Triton] = None
        self._triton_config: Optional[TritonConfig] = None
        self._is_new_instance: bool = False

    def get_input_schema(self) -> List[Tensor]:
        """
        Define the input tensor schema.

        Override this method to customize your input schema, or use the default
        VLM schema which supports images, videos, and JSON arguments.

        Default schema includes:
        - IMAGE: Multiple images (N_IMAGES, H, W, C) with uint8 dtype
        - VIDEO: Multiple videos (N_VIDEOS, T, H, W, C) with uint8 dtype
        - MEDIA_URLS: List of media URLs (N_URLS,) with dtype=bytes
        - MEDIA_MASK: Mask indicating media type (0: Image, 1: Video, 2: URL)
        - ARGS_JSON: Text arguments in JSON format as bytes

        Example:
            return [
                Tensor(name="IMAGE", dtype=np.uint8, shape=(224, 224, 3)),
                Tensor(name="TEXT", dtype=bytes, shape=(1,)),
            ]

        Returns:
            List of Tensor objects defining input schema
        """
        return [
            # Multiple Images
            Tensor(name="IMAGE", dtype=np.uint8,
                   shape=(-1, -1, -1, 3)),  # N_IMAGES, H, W, C
            # Multiple Videos
            Tensor(name="VIDEO", dtype=np.uint8,
                   shape=(-1, -1, -1, -1, 3)),  # N_VIDEOS, T, H, W, C
            # Media URL
            Tensor(name="MEDIA_URLS", dtype=bytes, shape=(-1,)),  # URLs list
            # Media Mask (0: Image, 1: Video, 2: URL, other: unknown)
            Tensor(name="MEDIA_MASK", dtype=np.uint8, shape=(-1,)),  # N_IMAGES + N_VIDEOS + N_URLS
            # Texts or Args in JSON format
            Tensor(name="ARGS_JSON", dtype=bytes, shape=(1,)),
        ]

    def get_output_schema(self) -> List[Tensor]:
        """
        Define the output tensor schema.

        Override this method to customize your output schema, or use the default
        schema which returns results in JSON format.

        Default schema includes:
        - RESULTS_JSON: Results in JSON format as bytes

        Example:
            return [
                Tensor(name="TEXT_OUT", dtype=bytes, shape=(1,)),
                Tensor(name="IMAGE_OUT", dtype=np.uint8, shape=(224, 224, 3)),
            ]

        Returns:
            List of Tensor objects defining output schema
        """
        return [
            # Results in JSON format
            Tensor(name="RESULTS_JSON", dtype=bytes, shape=(1,)),
        ]

    def inference_function(self, **inputs) -> Dict[str, np.ndarray]:
        """
        Inference function that processes inputs and returns outputs.

        This method is used when no inference_func is provided in __init__.
        Override this method in subclasses to implement your inference logic,
        or pass an inference_func directly to __init__ instead.

        The function will be decorated with @batch automatically, and the wrapper
        will:
          - Decode raw tensors into high-level media list and kwargs using decode_request_tensors()
          - Encode the high-level result back into tensors using encode_response()

        Example:
            def inference_function(self, media=None, **kwargs):
                # media is a list where each item is:
                #   - PIL.Image.Image  (image)
                #   - List[ PIL.Image ] (video frames)
                #   - URL string
                # kwargs contains arguments unpacked from ARGS_JSON
                # Your inference logic here
                results_json = b'{"result": "success"}'
                return {"RESULTS_JSON": np.array([results_json], dtype=np.object_)}

        Args:
            **inputs: Raw input tensors as keyword arguments (batched).
                Expected keys: IMAGE, VIDEO, MEDIA_URLS, MEDIA_MASK, ARGS_JSON

        Returns:
            Dictionary of output tensors. Must match get_output_schema().
            Default expects {"RESULTS_JSON": ...}

        Raises:
            NotImplementedError: If neither inference_func nor this method is implemented
        """
        # This method is only used when no inference_func is provided and
        # subclasses override it. The actual call is wrapped by
        # _create_inference_wrapper, which reconstructs media and kwargs.
        raise NotImplementedError(
            "Either provide inference_func in __init__ or override inference_function() method")

    def _create_inference_wrapper(self):
        """
        Create a wrapped inference function with @batch decorator.
        
        This method automatically applies the @batch decorator to either:
        - The inference_func passed to __init__, or
        - The inference_function() method
        
        Returns:
            Wrapped inference function with @batch decorator applied
        """
        # Get the inference function (either from __init__ or method)
        if self.inference_func is not None:
            original_func = self.inference_func
        else:
            original_func = self.inference_function

        # Apply @batch decorator
        @sample
        def wrapped_inference(**inputs):
            """
            Wrapped inference function used by Triton.

            Responsibilities:
              - Decode raw tensors into high-level media list and kwargs
              - Call the user-provided inference function with:
                    original_func(media=media, **kwargs)
              - Encode the high-level result back into tensors
            """
            media, extra_kwargs = decode_request_tensors(
                IMAGE=inputs.get("IMAGE"),
                VIDEO=inputs.get("VIDEO"),
                MEDIA_URLS=inputs.get("MEDIA_URLS"),
                MEDIA_MASK=inputs.get("MEDIA_MASK"),
                ARGS_JSON=inputs.get("ARGS_JSON"),
            )

            # Call the original function with high-level arguments
            result = original_func(media=media, **extra_kwargs)

            encoded_result = encode_response(result)
            return encoded_result

        return wrapped_inference

    def start(self):
        """Start the Triton server."""
        if self.running:
            raise RuntimeError("Server is already running")

        # Get or create Triton instance (shared across models with same config)
        self._triton, self._triton_config, self._is_new_instance = (
            TritonInstanceManager.get_or_create_triton(
                host=self.host,
                grpc_port=self.grpc_port if self.enable_grpc else None,
                http_port=self.http_port if self.enable_http else None,
                metrics_port=self.metrics_port,
                log_verbose=self.log_verbose,
                enable_grpc=self.enable_grpc,
                enable_http=self.enable_http,
            )
        )

        # Get input and output schemas
        inputs = self.get_input_schema()
        outputs = self.get_output_schema()

        # Create wrapped inference function
        infer_func = self._create_inference_wrapper()

        # Bind model to the Triton instance
        self._triton.bind(
            model_name=self.model_name,
            infer_func=infer_func,
            inputs=inputs,
            outputs=outputs,
            config=ModelConfig(batching=False),
            strict=True,
        )

        # Start serving (this will block)
        # Only call serve() if this is a new instance or if not already serving
        protocols = []
        if self.enable_grpc:
            protocols.append(f"gRPC {self.host}:{self.grpc_port}")
        if self.enable_http:
            protocols.append(f"HTTP {self.host}:{self.http_port}")
        
        # Atomically check and mark as serving
        should_serve = TritonInstanceManager.check_and_mark_serving(
            host=self.host,
            grpc_port=self.grpc_port if self.enable_grpc else None,
            http_port=self.http_port if self.enable_http else None,
            metrics_port=self.metrics_port,
        )
        
        if should_serve:
            if self._is_new_instance:
                print(f"Serving model '{self.model_name}' on {', '.join(protocols)} ...")
            else:
                print(f"Serving model '{self.model_name}' on existing Triton instance ({', '.join(protocols)}) ...")
            self.running = True
            self._triton.serve()
        else:
            print(f"Bound model '{self.model_name}' to existing Triton instance on {', '.join(protocols)} ...")
            print(f"Triton instance is already serving. Model '{self.model_name}' is ready.")
            self.running = True

    def stop(self):
        """Stop the Triton server."""
        if not self.running:
            return

        # Release reference to Triton instance
        if self._triton is not None:
            TritonInstanceManager.release_triton(
                host=self.host,
                grpc_port=self.grpc_port if self.enable_grpc else None,
                http_port=self.http_port if self.enable_http else None,
                metrics_port=self.metrics_port,
            )

        self.running = False
        self._triton = None
        self._triton_config = None
        self._is_new_instance = False

    def __enter__(self):
        """Context manager entry."""
        # Get or create Triton instance (shared across models with same config)
        self._triton, self._triton_config, self._is_new_instance = (
            TritonInstanceManager.get_or_create_triton(
                host=self.host,
                grpc_port=self.grpc_port if self.enable_grpc else None,
                http_port=self.http_port if self.enable_http else None,
                metrics_port=self.metrics_port,
                log_verbose=self.log_verbose,
                enable_grpc=self.enable_grpc,
                enable_http=self.enable_http,
            )
        )

        # Get input and output schemas
        inputs = self.get_input_schema()
        outputs = self.get_output_schema()

        # Create wrapped inference function
        infer_func = self._create_inference_wrapper()

        # Bind model to the Triton instance
        self._triton.bind(
            model_name=self.model_name,
            infer_func=infer_func,
            inputs=inputs,
            outputs=outputs,
            config=ModelConfig(batching=False),
            strict=True,
        )

        protocols = []
        if self.enable_grpc:
            protocols.append(f"gRPC {self.host}:{self.grpc_port}")
        if self.enable_http:
            protocols.append(f"HTTP {self.host}:{self.http_port}")
        
        if self._is_new_instance:
            print(f"Serving model '{self.model_name}' on {', '.join(protocols)} ...")
        else:
            print(f"Bound model '{self.model_name}' to existing Triton instance on {', '.join(protocols)} ...")
        
        self.running = True

        return self._triton

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Release reference to Triton instance
        if self._triton is not None:
            TritonInstanceManager.release_triton(
                host=self.host,
                grpc_port=self.grpc_port if self.enable_grpc else None,
                http_port=self.http_port if self.enable_http else None,
                metrics_port=self.metrics_port,
            )
        self.running = False
        self._triton = None
        self._triton_config = None
        self._is_new_instance = False
        # Triton context manager handles cleanup automatically
