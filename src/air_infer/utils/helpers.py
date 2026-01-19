"""
Helper utilities for client-server communication.
"""


def validate_request(request: dict) -> bool:
    """
    Validate a request dictionary.
    
    Args:
        request: Request dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(request, dict):
        return False
    # Add validation logic as needed
    return True


def validate_response(response: dict) -> bool:
    """
    Validate a response dictionary.
    
    Args:
        response: Response dictionary to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    # Add validation logic as needed
    return True

