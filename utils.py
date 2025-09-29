import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import colorsys


def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        if image is None:
            raise ValueError("Image is None")

        if len(image.shape) == 2: 
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            _, buffer = cv2.imencode('.jpg', image_bgr)
        else: 
            if image.shape[2] == 4: 
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3: 
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', image_rgb)
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")

        return base64.b64encode(buffer).decode('utf-8')

    except Exception as e:
        print(f"Error in image_to_base64: {e}")
        print(f"Image shape: {image.shape if image is not None else 'None'}")
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', black_image)
        return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is not None:
            return image
        else:
            raise ValueError("Failed to decode image")

    except Exception as e:
        print(f"Error in base64_to_image: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    try:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error in pil_to_cv2: {e}")
        return np.zeros((100, 100, 3), dtype=np.uint8)


def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    try:
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Error in cv2_to_pil: {e}")
        return Image.new('RGB', (100, 100), color='black')


def resize_image(image, width=None, height=None):
    """Resize image while maintaining aspect ratio"""
    if width is None and height is None:
        return image

    h, w = image.shape[:2]

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def normalize_image(image):
    """Normalize image pixel values to 0-1 range"""
    return image.astype(np.float32) / 255.0


def denormalize_image(image):
    """Convert normalized image back to 0-255 range"""
    return (image * 255).astype(np.uint8)


def check_image_size(image, min_width=100, min_height=100):
    """Check if image meets minimum size requirements"""
    h, w = image.shape[:2]
    return h >= min_height and w >= min_width


def get_image_info(image):
    """Get basic information about the image"""
    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    dtype = image.dtype
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'dtype': str(dtype)
    }


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color"""
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB"""
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def create_rainbow_gradient(width=200, height=30):
    """Create a rainbow gradient image"""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        hue = x / width
        r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
        gradient[:, x] = [b, g, r] 
    return gradient


def create_color_wheel(size=200):
    """Create a color wheel"""
    wheel = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    radius = size // 2

    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            distance = np.sqrt(dx * dx + dy * dy)

            if distance <= radius:
                angle = np.arctan2(dy, dx) % (2 * np.pi)
                hue = angle / (2 * np.pi)
                saturation = distance / radius
                value = 1.0

                r, g, b = hsv_to_rgb(hue, saturation, value)
                wheel[y, x] = [b, g, r] 

    return wheel


def apply_color_overlay(image, color, alpha=0.5):
    """Apply color overlay to image"""
    overlay = image.copy()
    overlay[:] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def create_brush_preview(size=50, color=(255, 255, 255), brush_size=10):
    """Create brush preview circle"""
    preview = np.zeros((size, size, 4), dtype=np.uint8)  
    center = size // 2
    cv2.circle(preview, (center, center), brush_size // 2, (*color, 255), -1)
    cv2.circle(preview, (center, center), brush_size // 2, (255, 255, 255, 255), 1)

    return preview
