"""
Triton client implementation using PyTriton.
"""
from typing import Dict, Any, Optional
from pytriton.client import ModelClient

from .base_client import BaseClient
from ..utils.vlm_utils import encode_request, decode_response_tensors


class VLMTritonClient(BaseClient):
    """
    Triton client for LLM/VLM inference using PyTriton.
    
    This class provides a framework for Triton-based communication.
    Override prepare_inputs() and process_outputs() to define your input/output schema.
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
            request: Dictionary of input tensors (e.g., {"IMAGE": np.array(...)})
            
        Returns:
            Dictionary of output tensors from the server
        """
        if not self._connected:
            self.connect()
        
        if self._client is None:
            raise RuntimeError("Client not initialized. Call connect() first.")
        
        # Use infer_sample for single sample inference
        # Override this method if you need batch inference (infer_batch)
        response = self._client.infer_sample(**request)
        return response
    
    def infer(self, **kwargs) -> Dict[str, Any]:
        """
        High-level inference method that handles input preparation and output processing.
        
        Args:
            **kwargs: Input data as keyword arguments
            
        Returns:
            Processed output dictionary
        """
        # Prepare inputs (override prepare_inputs() to customize)
        inputs = self.prepare_inputs(**kwargs)
        
        # Send request
        raw_outputs = self.send_request(inputs)
        
        # Process outputs (override process_outputs() to customize)
        processed_outputs = self.process_outputs(raw_outputs)
        
        return processed_outputs
    
    def prepare_inputs(
        self,
        media: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare input tensors from keyword arguments.
        
        This method processes media items (PIL Images, sequences of PIL Images, URLs)
        and converts them to the format expected by the server. All other keyword arguments
        are packed into ARGS_JSON.
        
        Args:
            media: List of media items [item1, item2, ...] where each item can be:
                - PIL.Image.Image (one image item)
                - Sequence of PIL.Image (one video item - sequence of frames)
                - String URL (http://, https://, etc.) (one URL item)
                - Tuple: (media_item, media_info) or (media_item,)
                - Dict: {'media': media_item, 'fps': fps, ...}
                Note: A single item will be automatically wrapped in a list.
            **kwargs: All other arguments will be packed into a dict and converted to JSON
        
        Returns:
            Dictionary of input tensors ready for inference with keys:
            - IMAGE: Concatenated image array (N_IMAGES, H, W, C) or empty array
            - VIDEO: Concatenated video array (N_VIDEOS, T, H, W, C) or empty array
            - MEDIA_URLS: URLs as bytes tensor (N_URLS,) or empty array
            - MEDIA_MASK: Mask indicating media type order (0: Image, 1: Video, 2: URL)
            - ARGS_JSON: JSON string containing all kwargs as bytes tensor
        """
        # Use utility function to encode request (media + kwargs) to tensors
        return encode_request(media=media, **kwargs)
    
    def process_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process output tensors from the server response.
        
        Override this method to define your output processing.
        Example:
            def process_outputs(self, outputs):
                processed = {}
                if "TEXT_OUT" in outputs:
                    text_bytes = np.squeeze(outputs["TEXT_OUT"]).item()
                    processed["text"] = text_bytes.decode("utf-8", errors="replace")
                if "IMAGE_OUT" in outputs:
                    processed["image"] = outputs["IMAGE_OUT"]
                return processed
        
        Args:
            outputs: Raw output dictionary from server
            
        Returns:
            Processed output dictionary or high-level object.
        """
        # Default implementation: decode RESULTS_JSON if present
        # Override this method to customize output processing further.
        return decode_response_tensors(outputs)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
