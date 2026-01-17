"""
Base server class for LLM/VLM inference handling.
"""
from typing import Dict, Any, List, Optional
import numpy as np


class BaseServer:
    """
    Base class for server implementations.
    
    This class provides the foundation for handling client requests.
    Subclasses should implement specific communication protocols.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Initialize the base server.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.running = False

    def start(self):
        """Start the server."""
        raise NotImplementedError("Subclasses must implement start()")

    def stop(self):
        """Stop the server."""
        raise NotImplementedError("Subclasses must implement stop()")

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming request.
        
        Args:
            request: Request dictionary with input tensors
            
        Returns:
            Response dictionary with output tensors
        """
        raise NotImplementedError("Subclasses must implement handle_request()")

    def get_input_schema(self) -> List[Any]:
        """
        Define the input tensor schema.
        
        Override this method to define your input schema.
        Should return a list of Tensor objects from pytriton.model_config.
        
        Example:
            from pytriton.model_config import Tensor
            return [
                Tensor(name="IMAGE", dtype=np.uint8, shape=(224, 224, 3)),
                Tensor(name="TEXT", dtype=bytes, shape=(1,)),
            ]
        
        Returns:
            List of Tensor objects defining input schema
        """
        raise NotImplementedError("Subclasses must implement get_input_schema()")

    def get_output_schema(self) -> List[Any]:
        """
        Define the output tensor schema.
        
        Override this method to define your output schema.
        Should return a list of Tensor objects from pytriton.model_config.
        
        Example:
            from pytriton.model_config import Tensor
            return [
                Tensor(name="TEXT_OUT", dtype=bytes, shape=(1,)),
                Tensor(name="IMAGE_OUT", dtype=np.uint8, shape=(224, 224, 3)),
            ]
        
        Returns:
            List of Tensor objects defining output schema
        """
        raise NotImplementedError("Subclasses must implement get_output_schema()")

    def inference_function(self, **inputs) -> Dict[str, np.ndarray]:
        """
        Inference function that processes inputs and returns outputs.
        
        Override this method to implement your inference logic.
        This function will be called with input tensors as keyword arguments.
        
        Example:
            def inference_function(self, IMAGE=None, TEXT=None):
                # Process inputs
                # IMAGE shape: (B, 224, 224, 3) when @batch is used
                # TEXT shape: (B, 1) with dtype=np.object_
                
                batch_size = IMAGE.shape[0] if IMAGE is not None else 1
                
                # Your inference logic here
                text_out = np.full((batch_size, 1), b"output", dtype=np.object_)
                image_out = np.ascontiguousarray(IMAGE).copy()
                
                return {"TEXT_OUT": text_out, "IMAGE_OUT": image_out}
        
        Args:
            **inputs: Input tensors as keyword arguments
            
        Returns:
            Dictionary of output tensors
        """
        raise NotImplementedError("Subclasses must implement inference_function()")
