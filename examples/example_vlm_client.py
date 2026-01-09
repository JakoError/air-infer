"""
Example client usage using TritonClient.

This example shows how to use TritonClient with media (images, videos, URLs)
and additional arguments that are automatically packed into JSON.
"""
from PIL import Image
from lm_inference_rpc.client import VLMTritonClient


def main():
    """Example usage of TritonClient."""
    # Create client instance
    client = VLMTritonClient(
        model_name="MyVLM",
        host="127.0.0.1",
        grpc_port=9100,
        lazy_init=False,
        timeout_s=10,
    )
    
    # Use context manager for automatic connection/disconnection
    with client:
        # Example 1: Single PIL Image with additional arguments
        # Note: media must be a list, even for a single item
        img = Image.new('RGB', (224, 224), color='red')
        result = client.infer(
            media=[img],  # List with one image item
            prompt="What is in this image?",
            temperature=0.7,
            max_tokens=100
        )
        print("Result with image:", result)
        
        # Example 2: Multiple media items (images + URL)
        img2 = Image.new('RGB', (224, 224), color='blue')
        result = client.infer(
            media=[img, img2, "https://example.com/image.jpg"],  # List with multiple items
            prompt="Describe these images",
            temperature=0.5
        )
        print("Result with multiple media:", result)
        
        # Example 3: Video (sequence of PIL Images)
        # Note: A sequence of PIL Images is ONE video item in the list
        video_frames = [Image.new('RGB', (224, 224), color='green') for _ in range(10)]
        result = client.infer(
            media=[video_frames],  # List with one video item (sequence of frames)
            prompt="What happens in this video?",
            max_tokens=200
        )
        print("Result with video:", result)
        
        # Example 4: Mixed media types
        # Each item in the list is one media item: image, video (sequence), or URL
        result = client.infer(
            media=[
                img,  # Image item
                video_frames,  # Video item (sequence of frames)
                "https://example.com/video.mp4",  # URL item
            ],
            prompt="Analyze all media",
            temperature=0.8,
            top_p=0.9
        )
        print("Result with mixed media:", result)


if __name__ == "__main__":
    main()

