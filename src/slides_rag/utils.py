import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def encode_to_base64(image_path: str) -> Optional[str]:
    """
    Encode an image file to base64 string representation.

    Args:
        image_path (str): Path to the image file to be encoded

    Returns:
        Optional[str]: Base64 encoded string of the image if successful, None otherwise

    Example:
        >>> encoded = encode_to_base64("path/to/image.jpg")
        >>> if encoded:
        >>>     print("Successfully encoded image")
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return

    except Exception as e:
        logging.error(f"Failed to encode image {image_path}: {str(e)}")
        return
