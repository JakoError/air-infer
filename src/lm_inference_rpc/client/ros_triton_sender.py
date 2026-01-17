"""
Triton client for ROS2 message transmission using PyTriton.
"""
from typing import Dict, Any, Optional
import numpy as np
from pytriton.client import ModelClient

from .base_client import BaseClient
from ..utils.ros_utils import serialize_ros_message, get_ros_message_type
from ..utils.verification import compute_message_checksum, verify_message_integrity


class ROSTritonSender(BaseClient):
    """
    Triton client for ROS2 message transmission using PyTriton.
    
    Serializes ROS2 messages and sends them to a Triton server.
    """

    def __init__(
            self,
            model_name: str,
            url: Optional[str] = None,
            host: str = "127.0.0.1",
            grpc_port: int = 9100,
            http_port: int = 8100,
            protocol: str = "grpc",
            lazy_init: bool = False,
            timeout_s: int = 10,
    ):
        """
        Initialize the Triton client.
        
        Args:
            model_name: Name of the model to connect to
            url: Full URL (e.g., "grpc://127.0.0.1:9100"). If provided, overrides host/port
            host: Server host address
            grpc_port: gRPC port number
            http_port: HTTP port number (if using HTTP protocol)
            protocol: Communication protocol ("grpc" or "http")
            lazy_init: If True, delay connection until first inference
            timeout_s: Timeout in seconds for waiting for model
        """
        super().__init__(host=host, port=grpc_port)
        self.model_name = model_name
        self.protocol = protocol
        self.lazy_init = lazy_init
        self.timeout_s = timeout_s

        # Build URL if not provided
        if url is None:
            if protocol == "grpc":
                url = f"grpc://{host}:{grpc_port}"
            else:
                url = f"http://{host}:{http_port}"

        self.url = url
        self._client: Optional[ModelClient] = None

    def connect(self):
        """Establish connection to the Triton server."""
        if self._client is None:
            self._client = ModelClient(
                self.url,
                self.model_name,
                lazy_init=self.lazy_init
            )
            if not self.lazy_init:
                self._client.wait_for_model(timeout_s=self.timeout_s)
            self._connected = True

    def disconnect(self):
        """Close connection to the Triton server."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._connected = False

    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the Triton server.
        
        Args:
            request: Dictionary of input tensors with MESSAGE_BYTES and MESSAGE_TYPE
            
        Returns:
            Dictionary of output tensors from the server
        """
        if not self._connected:
            self.connect()

        if self._client is None:
            raise RuntimeError("Client not initialized. Call connect() first.")

        response = self._client.infer_sample(**request)
        return response

    def send_message(
            self,
            message: Any,
            message_type: Optional[str] = None,
            verify: bool = False
    ) -> Dict[str, Any]:
        """
        Send a ROS2 message to the server.
        
        Args:
            message: ROS2 message object to send
            message_type: ROS2 message type string (auto-detected if None)
            verify: If True, compute checksum and verify against server response (default: False)
            
        Returns:
            Dictionary with processed response (e.g., {"received": True}).
            If verify=True and server provides checksum, also includes:
            - "checksum": Server's checksum
            - "checksum_valid": Whether client and server checksums match
            - "client_checksum": Client's computed checksum
        """
        inputs = self.prepare_inputs(message=message, message_type=message_type)

        # Compute client-side checksum if verification is enabled
        client_checksum = None
        if verify:
            msg_bytes = inputs["MESSAGE_BYTES"][0]
            if isinstance(msg_bytes, np.ndarray):
                msg_bytes = msg_bytes.item()
            client_checksum = compute_message_checksum(msg_bytes)

        raw_outputs = self.send_request(inputs)
        processed = self.process_outputs(raw_outputs)

        # Verify checksums if both client and server provided them
        if verify and client_checksum and "checksum" in processed:
            server_checksum = processed["checksum"]
            # Extract message bytes for verification
            msg_bytes = inputs["MESSAGE_BYTES"][0]
            if isinstance(msg_bytes, np.ndarray):
                msg_bytes = msg_bytes.item()

            is_valid, computed = verify_message_integrity(msg_bytes, server_checksum)
            processed["checksum_valid"] = is_valid
            processed["client_checksum"] = client_checksum
            if not is_valid:
                processed[
                    "verification_error"] = f"Checksum mismatch: client={client_checksum[:16]}... server={server_checksum[:16]}..."

        return processed

    def prepare_inputs(
            self,
            message: Optional[Any] = None,
            message_type: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare input tensors from a ROS2 message.
        
        Serializes the ROS2 message to bytes and includes the message type.
        
        Args:
            message: ROS2 message object to serialize
            message_type: ROS2 message type string (e.g., "std_msgs/String").
                         If None, will be inferred from the message object.
            **kwargs: Additional arguments (not used, but kept for compatibility)
        
        Returns:
            Dictionary with keys:
            - MESSAGE_TYPE: Message type string as bytes tensor
            - MESSAGE_BYTES: Serialized message bytes as numpy array
        """
        if message is None:
            raise ValueError("message is required")

        # Get message type if not provided
        if message_type is None:
            message_type = get_ros_message_type(message)

        # Serialize message to bytes
        msg_bytes = serialize_ros_message(message)

        return {
            "MESSAGE_TYPE": np.array([message_type.encode('utf-8')], dtype=np.object_),
            "MESSAGE_BYTES": np.array([msg_bytes], dtype=np.object_),
        }

    def process_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process output tensors from the server response.
        
        Args:
            outputs: Raw output dictionary from server
            
        Returns:
            Dictionary with processed outputs. Default expects RECEIVE_FLAG.
            If verification is enabled, also includes verification results.
        """
        processed = {}
        if "RECEIVE_FLAG" in outputs:
            flag = outputs["RECEIVE_FLAG"]
            if isinstance(flag, np.ndarray):
                processed["received"] = bool(np.squeeze(flag).item())
            else:
                processed["received"] = bool(flag)

        # Process checksum if present
        if "MESSAGE_CHECKSUM" in outputs:
            checksum_arr = outputs["MESSAGE_CHECKSUM"]
            if isinstance(checksum_arr, np.ndarray):
                checksum_bytes = checksum_arr[0]
                if isinstance(checksum_bytes, np.ndarray):
                    checksum_bytes = checksum_bytes.item()
                processed["checksum"] = checksum_bytes.decode('utf-8') if isinstance(checksum_bytes, bytes) else str(
                    checksum_bytes)

        return processed

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
