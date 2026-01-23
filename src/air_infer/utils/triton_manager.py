"""
Triton instance manager for sharing Triton instances across multiple models.

This module provides a singleton-like manager that allows multiple servers
to bind their models to the same Triton instance, identified by host and ports.
"""
from typing import Optional, Dict, Tuple
from threading import Lock, Event
from pytriton.triton import Triton, TritonConfig


class TritonInstanceManager:
    """
    Manages Triton instances to allow multiple models to share the same instance.
    
    Instances are identified by a unique key based on host and port configuration.
    Multiple servers can bind their models to the same Triton instance.
    """
    
    _instances: Dict[str, Tuple[Triton, TritonConfig, int, Event]] = {}
    _lock = Lock()
    
    @classmethod
    def _get_instance_key(
        cls,
        host: str,
        grpc_port: Optional[int],
        http_port: Optional[int],
        metrics_port: int,
    ) -> str:
        """
        Generate a unique key for a Triton instance configuration.
        
        Args:
            host: Server host address
            grpc_port: gRPC port (None if disabled)
            http_port: HTTP port (None if disabled)
            metrics_port: Metrics port
            
        Returns:
            Unique string key identifying this configuration
        """
        return f"{host}:{grpc_port}:{http_port}:{metrics_port}"
    
    @classmethod
    def get_or_create_triton(
        cls,
        host: str,
        grpc_port: Optional[int],
        http_port: Optional[int],
        metrics_port: int,
        log_verbose: int = 1,
        enable_grpc: bool = True,
        enable_http: bool = True,
    ) -> Tuple[Triton, TritonConfig, bool]:
        """
        Get an existing Triton instance or create a new one.
        
        If an instance with the same configuration already exists, returns it.
        Otherwise, creates a new instance and stores it.
        
        Args:
            host: Server host address
            grpc_port: gRPC port number (None if disabled)
            http_port: HTTP port number (None if disabled)
            metrics_port: Metrics port number
            log_verbose: Log verbosity level
            enable_grpc: Whether gRPC is enabled
            enable_http: Whether HTTP is enabled
            
        Returns:
            Tuple of (Triton instance, TritonConfig, is_new_instance)
            is_new_instance is True if this is a newly created instance (first one)
        """
        instance_key = cls._get_instance_key(host, grpc_port, http_port, metrics_port)
        
        with cls._lock:
            if instance_key in cls._instances:
                triton, config, ref_count, serving_event = cls._instances[instance_key]
                # Increment reference count
                cls._instances[instance_key] = (triton, config, ref_count + 1, serving_event)
                return triton, config, False
            
            # Create new instance
            config = TritonConfig(
                grpc_address=host if enable_grpc else None,
                http_address=host if enable_http else None,
                metrics_address=host,
                grpc_port=grpc_port if enable_grpc else None,
                http_port=http_port if enable_http else None,
                metrics_port=metrics_port,
                log_verbose=log_verbose,
            )
            
            triton = Triton(config=config)
            serving_event = Event()  # Track if serve() has been called
            
            # Store with reference count of 1
            cls._instances[instance_key] = (triton, config, 1, serving_event)
            
            return triton, config, True
    
    @classmethod
    def mark_serving(
        cls,
        host: str,
        grpc_port: Optional[int],
        http_port: Optional[int],
        metrics_port: int,
    ):
        """
        Mark that serve() has been called for this instance.
        
        Args:
            host: Server host address
            grpc_port: gRPC port number (None if disabled)
            http_port: HTTP port number (None if disabled)
            metrics_port: Metrics port number
        """
        instance_key = cls._get_instance_key(host, grpc_port, http_port, metrics_port)
        
        with cls._lock:
            if instance_key in cls._instances:
                triton, config, ref_count, serving_event = cls._instances[instance_key]
                serving_event.set()
    
    @classmethod
    def check_and_mark_serving(
        cls,
        host: str,
        grpc_port: Optional[int],
        http_port: Optional[int],
        metrics_port: int,
    ) -> bool:
        """
        Atomically check if serving and mark as serving if not already serving.
        
        This method is thread-safe and prevents race conditions when multiple
        servers try to start serving the same instance.
        
        Args:
            host: Server host address
            grpc_port: gRPC port number (None if disabled)
            http_port: HTTP port number (None if disabled)
            metrics_port: Metrics port number
            
        Returns:
            True if this call marked it as serving (should call serve()),
            False if it was already serving (should not call serve())
        """
        instance_key = cls._get_instance_key(host, grpc_port, http_port, metrics_port)
        
        with cls._lock:
            if instance_key not in cls._instances:
                return False
            
            triton, config, ref_count, serving_event = cls._instances[instance_key]
            
            if serving_event.is_set():
                return False  # Already serving
            
            serving_event.set()
            return True  # We marked it as serving
    
    @classmethod
    def is_serving(
        cls,
        host: str,
        grpc_port: Optional[int],
        http_port: Optional[int],
        metrics_port: int,
    ) -> bool:
        """
        Check if serve() has been called for this instance.
        
        Args:
            host: Server host address
            grpc_port: gRPC port number (None if disabled)
            http_port: HTTP port number (None if disabled)
            metrics_port: Metrics port number
            
        Returns:
            True if serve() has been called, False otherwise
        """
        instance_key = cls._get_instance_key(host, grpc_port, http_port, metrics_port)
        
        with cls._lock:
            if instance_key in cls._instances:
                _, _, _, serving_event = cls._instances[instance_key]
                return serving_event.is_set()
            return False
    
    @classmethod
    def release_triton(
        cls,
        host: str,
        grpc_port: Optional[int],
        http_port: Optional[int],
        metrics_port: int,
    ) -> bool:
        """
        Release a reference to a Triton instance.
        
        When the reference count reaches zero, the instance is removed from tracking.
        Note: The actual Triton instance cleanup is handled by its context manager.
        
        Args:
            host: Server host address
            grpc_port: gRPC port number (None if disabled)
            http_port: HTTP port number (None if disabled)
            metrics_port: Metrics port number
            
        Returns:
            True if instance was removed (ref count reached 0), False otherwise
        """
        instance_key = cls._get_instance_key(host, grpc_port, http_port, metrics_port)
        
        with cls._lock:
            if instance_key not in cls._instances:
                return False
            
            triton, config, ref_count, serving_event = cls._instances[instance_key]
            
            if ref_count <= 1:
                # Last reference, remove from tracking
                del cls._instances[instance_key]
                return True
            else:
                # Decrement reference count
                cls._instances[instance_key] = (triton, config, ref_count - 1, serving_event)
                return False
    
    @classmethod
    def get_instance_count(cls) -> int:
        """
        Get the number of active Triton instances.
        
        Returns:
            Number of unique Triton instances currently tracked
        """
        with cls._lock:
            return len(cls._instances)
    
    @classmethod
    def clear_all(cls):
        """
        Clear all tracked instances (mainly for testing).
        
        Warning: This does not stop or cleanup the Triton instances themselves.
        """
        with cls._lock:
            cls._instances.clear()
