import cv2
import numpy as np
import os
from inference_sdk import InferenceHTTPClient


class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)


class ImageTransformer:
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.face_detection_enabled = False
        self.face_zoom_enabled = False
        self.drawing_mode = False
        self.brush_color = (255, 255, 255)
        self.brush_size = 5
        self.last_mouse_pos = None

        # Initialize Roboflow client for face detection
        try:
            self.roboflow_client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key="3xHadLJ7GhY5tMP7zbdE"
            )
        except:
            self.roboflow_client = None

    def set_image(self, image):
        self.original_image = image.copy()
        self.current_image = image.copy()
        return self.current_image

    def reset(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        return self.current_image

    def translate(self, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        self.current_image = cv2.warpAffine(self.current_image, M,
                                            (self.current_image.shape[1], self.current_image.shape[0]))
        return self.current_image

    def rotate(self, angle, center=None, scale=1.0):
        (h, w) = self.current_image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.current_image = cv2.warpAffine(self.current_image, M, (w, h))
        return self.current_image

    def resize(self, width=None, height=None, inter=cv2.INTER_AREA):
        if width is None and height is None:
            return self.current_image

        if width is None:
            r = height / float(self.current_image.shape[0])
            dim = (int(self.current_image.shape[1] * r), height)
        else:
            r = width / float(self.current_image.shape[1])
            dim = (width, int(self.current_image.shape[0] * r))

        self.current_image = cv2.resize(self.current_image, dim, interpolation=inter)
        return self.current_image

    def flip(self, code):
        self.current_image = cv2.flip(self.current_image, code)
        return self.current_image

    def crop(self, y1, y2, x1, x2):
        # Ensure coordinates are within bounds
        h, w = self.current_image.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        self.current_image = self.current_image[y1:y2, x1:x2]
        return self.current_image

    def arithmetic_operations(self, operation, value):
        if operation == "add":
            M = np.ones(self.current_image.shape, dtype="uint8") * value
            self.current_image = cv2.add(self.current_image, M)
        elif operation == "subtract":
            M = np.ones(self.current_image.shape, dtype="uint8") * value
            self.current_image = cv2.subtract(self.current_image, M)
        elif operation == "multiply":
            self.current_image = cv2.multiply(self.current_image, np.array([value]))
        elif operation == "divide":
            self.current_image = cv2.divide(self.current_image, np.array([value]))
        return self.current_image

    def bitwise_operations(self, operation, shape_type="rectangle", coordinates=None):
        if shape_type == "rectangle":
            mask = np.zeros(self.current_image.shape[:2], dtype="uint8")
            cv2.rectangle(mask, coordinates[0], coordinates[1], 255, -1)
        elif shape_type == "circle":
            mask = np.zeros(self.current_image.shape[:2], dtype="uint8")
            cv2.circle(mask, coordinates[0], coordinates[1], 255, -1)
        elif shape_type == "polygon":
            mask = np.zeros(self.current_image.shape[:2], dtype="uint8")
            cv2.fillPoly(mask, [np.array(coordinates)], 255)

        if operation == "and":
            self.current_image = cv2.bitwise_and(self.current_image, self.current_image, mask=mask)
        elif operation == "or":
            self.current_image = cv2.bitwise_or(self.current_image, self.current_image, mask=mask)
        elif operation == "xor":
            self.current_image = cv2.bitwise_xor(self.current_image, self.current_image, mask=mask)
        elif operation == "not":
            self.current_image = cv2.bitwise_not(self.current_image)

        return self.current_image

    def create_shapes(self, shape_type, coordinates, color=(255, 255, 255), thickness=-1):
        # Create a copy to draw on
        temp_image = self.current_image.copy()

        if shape_type == "rectangle":
            cv2.rectangle(temp_image, coordinates[0], coordinates[1], color, thickness)
        elif shape_type == "circle":
            cv2.circle(temp_image, coordinates[0], coordinates[1], color, thickness)
        elif shape_type == "line":
            cv2.line(temp_image, coordinates[0], coordinates[1], color, thickness)
        elif shape_type == "polygon":
            cv2.fillPoly(temp_image, [np.array(coordinates)], color)
        elif shape_type == "text":
            cv2.putText(temp_image, coordinates[0], coordinates[1],
                        cv2.FONT_HERSHEY_SIMPLEX, coordinates[2], color, thickness)

        self.current_image = temp_image
        return self.current_image

    def color_operations(self, operation, parameters=None):
        if operation == "grayscale":
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            if len(self.current_image.shape) == 2:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)

        elif operation == "hsv":
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)

        elif operation == "lab":
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)

        elif operation == "negative":
            self.current_image = 255 - self.current_image

        elif operation == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            self.current_image = cv2.transform(self.current_image, kernel)
            self.current_image = np.clip(self.current_image, 0, 255)

        elif operation == "warm":
            self.current_image[:, :, 0] = np.clip(self.current_image[:, :, 0] * 0.9, 0, 255)  # Blue
            self.current_image[:, :, 2] = np.clip(self.current_image[:, :, 2] * 1.1, 0, 255)  # Red

        elif operation == "cool":
            self.current_image[:, :, 0] = np.clip(self.current_image[:, :, 0] * 1.1, 0, 255)  # Blue
            self.current_image[:, :, 2] = np.clip(self.current_image[:, :, 2] * 0.9, 0, 255)  # Red

        elif operation == "adjust_brightness":
            value = parameters.get('value', 0)
            # Reset to original first to avoid stacking
            temp_image = self.original_image.copy() if self.original_image is not None else self.current_image.copy()
            self.current_image = cv2.convertScaleAbs(temp_image, alpha=1, beta=value)

        elif operation == "adjust_contrast":
            value = parameters.get('value', 0)
            # Reset to original first to avoid stacking
            temp_image = self.original_image.copy() if self.original_image is not None else self.current_image.copy()
            alpha = 1 + value / 100.0
            self.current_image = cv2.convertScaleAbs(temp_image, alpha=alpha, beta=0)

        elif operation == "adjust_saturation":
            value = parameters.get('value', 0)
            # Reset to original first to avoid stacking
            temp_image = self.original_image.copy() if self.original_image is not None else self.current_image.copy()
            hsv = cv2.cvtColor(temp_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.add(s, value)
            final_hsv = cv2.merge((h, s, v))
            self.current_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        elif operation == "adjust_hue":
            value = parameters.get('value', 0)
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h = cv2.add(h, value)
            final_hsv = cv2.merge((h, s, v))
            self.current_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        elif operation == "color_overlay":
            # Add color overlay
            color = parameters.get('color', [255, 0, 0])
            alpha = parameters.get('alpha', 0.5)
            overlay = self.current_image.copy()
            overlay[:] = color
            self.current_image = cv2.addWeighted(self.current_image, 1 - alpha, overlay, alpha, 0)

        return self.current_image

    def filter_operations(self, operation, parameters=None):
        if operation == "blur":
            kernel_size = parameters.get('kernel_size', 15)
            self.current_image = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)

        elif operation == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)

        elif operation == "emboss":
            kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)

        elif operation == "gaussian_blur":
            kernel_size = parameters.get('kernel_size', 15)
            self.current_image = cv2.GaussianBlur(self.current_image, (kernel_size, kernel_size), 0)

        elif operation == "median_blur":
            kernel_size = parameters.get('kernel_size', 15)
            self.current_image = cv2.medianBlur(self.current_image, kernel_size)

        elif operation == "bilateral_filter":
            self.current_image = cv2.bilateralFilter(self.current_image, 9, 75, 75)

        return self.current_image

    def edge_detection(self, method, parameters=None):
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        if method == "canny":
            threshold1 = parameters.get('threshold1', 100)
            threshold2 = parameters.get('threshold2', 200)
            edges = cv2.Canny(gray, threshold1, threshold2)

        elif method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = cv2.magnitude(sobelx, sobely)
            edges = cv2.convertScaleAbs(edges)

        elif method == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = cv2.convertScaleAbs(edges)

        elif method == "prewitt":
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            edges = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)

        elif method == "roberts":
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            robertsx = cv2.filter2D(gray, -1, kernelx)
            robertsy = cv2.filter2D(gray, -1, kernely)
            edges = cv2.addWeighted(robertsx, 0.5, robertsy, 0.5, 0)

        self.current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def morphological_operations(self, operation, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation == "erode":
            self.current_image = cv2.erode(self.current_image, kernel, iterations=1)
        elif operation == "dilate":
            self.current_image = cv2.dilate(self.current_image, kernel, iterations=1)
        elif operation == "open":
            self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, kernel)
        elif operation == "gradient":
            self.current_image = cv2.morphologyEx(self.current_image, cv2.MORPH_GRADIENT, kernel)

        return self.current_image

    def threshold_operations(self, method, threshold=127, max_value=255):
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

        if method == "binary":
            _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
        elif method == "binary_inv":
            _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY_INV)
        elif method == "trunc":
            _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_TRUNC)
        elif method == "tozero":
            _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_TOZERO)
        elif method == "tozero_inv":
            _, thresh = cv2.threshold(gray, threshold, max_value, cv2.THRESH_TOZERO_INV)
        elif method == "otsu":
            _, thresh = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive_mean":
            thresh = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method == "adaptive_gaussian":
            thresh = cv2.adaptiveThreshold(gray, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

        self.current_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return self.current_image

    def detect_faces(self):
        try:
            # Используем Haar cascade для детекции лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            # Конвертируем в grayscale для лучшей детекции
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            # Детектируем лица с улучшенными параметрами
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # Создаем копию для рисования
            result_image = self.current_image.copy()

            # Рисуем прямоугольники вокруг лиц
            for i, (x, y, w, h) in enumerate(faces):
                # Рисуем зеленый прямоугольник
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Добавляем текст с номером лица
                cv2.putText(result_image, f"Face {i + 1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(f"Detected {len(faces)} faces")
            self.current_image = result_image
            return self.current_image

        except Exception as e:
            print(f"Face detection error: {e}")
            return self.current_image

    def zoom_face(self):
        try:
            # Создаем копию текущего изображения
            image_copy = self.current_image.copy()

            # Используем детектор лиц
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

            # Детектируем лица
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                # Берем самое большое лицо
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                x, y, w, h = faces[0]

                print(f"Face detected at: x={x}, y={y}, w={w}, h={h}")

                # Добавляем отступы (30% от размера лица)
                padding_x = int(w * 0.3)
                padding_y = int(h * 0.3)

                # Рассчитываем координаты обрезки
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(image_copy.shape[1], x + w + padding_x)
                y2 = min(image_copy.shape[0], y + h + padding_y)

                print(f"Cropping to: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Обрезаем изображение до области лица
                cropped_face = image_copy[y1:y2, x1:x2]

                # Если обрезанная область не пустая
                if cropped_face.size > 0:
                    self.current_image = cropped_face
                    print(f"Zoomed face size: {self.current_image.shape}")
                else:
                    print("Cropped face is empty, using original")

            else:
                print("No faces detected for zoom")

            return self.current_image

        except Exception as e:
            print(f"Face zoom error: {e}")
            import traceback
            traceback.print_exc()
            return self.current_image

    def toggle_face_detection(self):
        self.face_detection_enabled = not self.face_detection_enabled
        return self.face_detection_enabled

    def toggle_face_zoom(self):
        self.face_zoom_enabled = not self.face_zoom_enabled
        return self.face_zoom_enabled

    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        return self.drawing_mode

    def set_brush_color(self, color):
        self.brush_color = color

    def set_brush_size(self, size):
        self.brush_size = size

    def draw_on_image(self, start_pos, end_pos):
        if not self.drawing_mode:
            return self.current_image

        # Draw line from start to end position
        cv2.line(self.current_image, start_pos, end_pos, self.brush_color, self.brush_size)
        return self.current_image

    def histogram_equalization(self):
        # Convert to YUV color space
        yuv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2YUV)
        # Equalize the histogram of the Y channel
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        # Convert back to BGR
        self.current_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return self.current_image

    def noise_operations(self, operation, parameters=None):
        if operation == "gaussian_noise":
            mean = parameters.get('mean', 0)
            std = parameters.get('std', 25)
            noise = np.random.normal(mean, std, self.current_image.shape).astype(np.uint8)
            self.current_image = cv2.add(self.current_image, noise)

        elif operation == "salt_pepper_noise":
            amount = parameters.get('amount', 0.05)
            s_vs_p = parameters.get('s_vs_p', 0.5)

            # Salt mode
            num_salt = np.ceil(amount * self.current_image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.current_image.shape]
            self.current_image[coords[0], coords[1], :] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * self.current_image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.current_image.shape]
            self.current_image[coords[0], coords[1], :] = 0

        elif operation == "poisson_noise":
            noise = np.random.poisson(self.current_image).astype(np.uint8)
            self.current_image = noise

        elif operation == "speckle_noise":
            noise = np.random.randn(*self.current_image.shape).astype(np.uint8)
            self.current_image = cv2.add(self.current_image, noise)

        return self.current_image

    def get_current_image(self):
        return self.current_image

    def get_preview_with_shape(self, shape_type, coordinates, color=(255, 255, 255), thickness=2):
        """Get preview image with shape overlay (for real-time preview)"""
        if self.current_image is None:
            return None

        preview = self.current_image.copy()

        if shape_type == "rectangle":
            cv2.rectangle(preview, coordinates[0], coordinates[1], color, thickness)
        elif shape_type == "circle":
            cv2.circle(preview, coordinates[0], coordinates[1], color, thickness)
        elif shape_type == "line":
            cv2.line(preview, coordinates[0], coordinates[1], color, thickness)

        return preview

    def get_preview_with_crop(self, x1, y1, x2, y2):
        """Get preview image with crop overlay"""
        if self.current_image is None:
            return None

        preview = self.current_image.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return preview

    def get_preview_with_rotation(self, angle):
        """Get preview image with rotation"""
        if self.current_image is None:
            return None

        (h, w) = self.current_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        preview = cv2.warpAffine(self.current_image, M, (w, h))
        return preview


# Global transformer instance
transformer = ImageTransformer()