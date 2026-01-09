"""
Base client class for LLM/VLM inference communication.
"""
from typing import Dict, Any, Optional


class BaseClient:
    """
    Base class for client implementations.
    
    This class provides the foundation for client-server communication.
    Subclasses should implement specific communication protocols.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000):
        """
        Initialize the base client.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self._connected = False
    
    def connect(self):
        """Establish connection to the server."""
        raise NotImplementedError("Subclasses must implement connect()")
    
    def disconnect(self):
        """Close connection to the server."""
        raise NotImplementedError("Subclasses must implement disconnect()")
    
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the server.
        
        Args:
            request: Request dictionary with input tensors
            
        Returns:
            Response dictionary with output tensors
        """
        raise NotImplementedError("Subclasses must implement send_request()")
    
    def prepare_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Prepare input tensors from keyword arguments.
        
        Override this method to define your input schema.
        
        Args:
            **kwargs: Input data as keyword arguments
            
        Returns:
            Dictionary of input tensors ready for inference
        """
        raise NotImplementedError("Subclasses must implement prepare_inputs()")
    
    def process_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process output tensors from the server response.
        
        Override this method to define your output processing.
        
        Args:
            outputs: Raw output dictionary from server
            
        Returns:
            Processed output dictionary
        """
        raise NotImplementedError("Subclasses must implement process_outputs()")

