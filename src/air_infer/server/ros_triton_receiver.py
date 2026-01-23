"""
Triton server for ROS2 message reception using PyTriton.
"""
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from pytriton.decorators import sample
from pytriton.model_config import Tensor, ModelConfig
from pytriton.triton import Triton, TritonConfig

from .base_server import BaseServer
from ..utils.ros_utils import deserialize_ros_message
from ..utils.verification import compute_message_checksum
from ..utils.triton_manager import TritonInstanceManager


class ROSTritonReceiver(BaseServer):
    """
    Triton server for ROS2 message reception using PyTriton.

    Deserializes ROS2 messages from bytes and calls user-defined handler.

    Usage:
        def my_handler(message):
            # Process ROS2 message
            return True  # Return True to indicate success

        server = ROSTritonReceiver(
            model_name="MyModel",
            inference_func=my_handler
        )
    """

    def __init__(
            self,
            model_name: str,
            inference_func: Optional[Callable[...,
            Dict[str, np.ndarray]]] = None,
            host: str = "127.0.0.1",
            grpc_port: int = 9100,
            http_port: int = 8100,
            metrics_port: int = 8101,
            log_verbose: int = 1,
            enable_grpc: bool = True,
            enable_http: bool = True,
            enable_verification: bool = False,
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
            enable_verification: If True, include message checksum in response for verification (default: False)
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
        self.enable_verification = enable_verification

        self._triton: Optional[Triton] = None
        self._triton_config: Optional[TritonConfig] = None
        self._is_new_instance: bool = False

    def get_input_schema(self) -> List[Tensor]:
        """
        Define the input tensor schema for ROS2 messages.

        Returns:
            List of Tensor objects with:
            - MESSAGE_BYTES: Serialized ROS2 message bytes
            - MESSAGE_TYPE: Message type string as bytes
        """
        return [
            Tensor(name="MESSAGE_TYPE", dtype=bytes, shape=(1,)),
            Tensor(name="MESSAGE_BYTES", dtype=bytes, shape=(1,)),
        ]

    def get_output_schema(self) -> List[Tensor]:
        """
        Define the output tensor schema.

        Returns:
            List of Tensor objects with RECEIVE_FLAG indicating success.
            If enable_verification is True, also includes MESSAGE_CHECKSUM.
        """
        outputs = [
            Tensor(name="RECEIVE_FLAG", dtype=np.bool_, shape=(1,)),
        ]
        if self.enable_verification:
            outputs.append(
                Tensor(name="MESSAGE_CHECKSUM", dtype=bytes, shape=(1,))
            )
        return outputs

    def inference_function(self, message: Any) -> bool:
        """
        Inference function that processes ROS2 messages.

        Override this method in subclasses, or pass inference_func to __init__.
        The wrapper will deserialize the message before calling this function.

        Args:
            message: Deserialized ROS2 message object

        Returns:
            True to indicate successful processing
        """
        return True

    def _create_inference_wrapper(self):
        """
        Create a wrapped inference function with @sample decorator.

        Deserializes ROS2 messages from bytes before calling user function.

        Returns:
            Wrapped inference function with @sample decorator applied
        """
        if self.inference_func is not None:
            original_func = self.inference_func
        else:
            original_func = self.inference_function

        @sample
        def wrapped_inference(**inputs):
            """
            Wrapped inference function that deserializes ROS2 messages.
            """
            # Extract message bytes and type
            msg_type_arr = inputs.get("MESSAGE_TYPE")
            msg_bytes_arr = inputs.get("MESSAGE_BYTES")

            if msg_type_arr is None or msg_bytes_arr is None:
                raise ValueError("MESSAGE_TYPE and MESSAGE_BYTES are required")

            # Extract type string and bytes
            msg_type_bytes = msg_type_arr[0]
            if isinstance(msg_type_bytes, np.ndarray):
                msg_type_bytes = msg_type_bytes.item()
            msg_type_str = msg_type_bytes.decode('utf-8')

            msg_bytes = msg_bytes_arr[0]
            if isinstance(msg_bytes, np.ndarray):
                msg_bytes = msg_bytes.item()

            # Deserialize ROS2 message
            message = deserialize_ros_message(msg_type_str, msg_bytes)

            # Call user function with deserialized message
            result = original_func(message)

            # Prepare response
            response = {"RECEIVE_FLAG": np.array([bool(result)], dtype=bool)}

            # Add checksum for verification if enabled
            if self.enable_verification:
                checksum = compute_message_checksum(msg_bytes)
                response["MESSAGE_CHECKSUM"] = np.array([checksum.encode('utf-8')], dtype=np.object_)

            return response

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
            print(f"Serving receiver '{self.model_name}' on {', '.join(protocols)} ...")
        else:
            print(f"Bound receiver '{self.model_name}' to existing Triton instance on {', '.join(protocols)} ...")
        
        self.running = True

        return self

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
