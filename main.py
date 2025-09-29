from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import json
import os
from image_processor import ImageTransformer, SimplePreprocessor
from logo_classifier import LogoClassifier
from utils import base64_to_image, image_to_base64

app = Flask(__name__)

# Initialize classifiers and processors
classifier = LogoClassifier("keras_Model.h5", "labels.txt")
preprocessor = SimplePreprocessor(32, 32)
transformer = ImageTransformer()

# Global settings
ENABLE_CLASSIFICATION = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    global ENABLE_CLASSIFICATION
    try:
        data = request.get_json()
        image_data = data['image']
        operation = data['operation']
        parameters = data.get('parameters', {})

        print(f"Processing operation: {operation} with parameters: {parameters}")

        # Convert base64 to OpenCV image
        if image_data.startswith('data:image'):
            image = base64_to_image(image_data.split(',')[1])
        else:
            image = base64_to_image(image_data)

        # Reset transformer for new image or reset operation
        if operation == 'original' or operation == 'reset_chain' or transformer.current_image is None:
            transformer.set_image(image)

        # Apply operations
        processed_image = apply_image_operations(operation, parameters)

        # Делаем предсказание ТОЛЬКО если классификация включена И это не операции рисования/фильтров
        prediction = None
        if ENABLE_CLASSIFICATION and operation not in ['detect_faces', 'zoom_face_only', 'draw_rectangle', 'draw_circle', 'draw_line', 'draw_polygon', 'draw_text']:
            try:
                prediction_image = processed_image.copy()
                class_name, confidence = classifier.predict(prediction_image)
                prediction = {
                    'class_name': class_name,
                    'confidence': confidence
                }
                print(f"Prediction: {class_name} with confidence: {confidence}")
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = {
                    'class_name': 'Unknown',
                    'confidence': 0.0
                }

        # Convert back to base64
        processed_base64 = image_to_base64(processed_image)

        return jsonify({
            'success': True,
            'processed_image': processed_base64,
            'prediction': prediction
        })

    except Exception as e:
        print(f"Error in process_image: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/detect_faces_only', methods=['POST'])
def detect_faces_only():
    try:
        data = request.get_json()
        image_data = data['image']

        # Convert base64 to OpenCV image
        if image_data.startswith('data:image'):
            image = base64_to_image(image_data.split(',')[1])
        else:
            image = base64_to_image(image_data)

        transformer.set_image(image)
        processed_image = transformer.detect_faces()
        processed_base64 = image_to_base64(processed_image)

        return jsonify({
            'success': True,
            'processed_image': processed_base64
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/zoom_face_only', methods=['POST'])
def zoom_face_only():
    try:
        data = request.get_json()
        image_data = data['image']

        # Convert base64 to OpenCV image
        if image_data.startswith('data:image'):
            image = base64_to_image(image_data.split(',')[1])
        else:
            image = base64_to_image(image_data)

        transformer.set_image(image)
        processed_image = transformer.zoom_face()
        processed_base64 = image_to_base64(processed_image)

        return jsonify({
            'success': True,
            'processed_image': processed_base64
        })

    except Exception as e:
        print(f"Zoom face error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/toggle_classification', methods=['POST'])
def toggle_classification():
    global ENABLE_CLASSIFICATION
    ENABLE_CLASSIFICATION = not ENABLE_CLASSIFICATION
    return jsonify({
        'success': True,
        'enabled': ENABLE_CLASSIFICATION
    })


@app.route('/toggle_face_detection', methods=['POST'])
def toggle_face_detection():
    enabled = transformer.toggle_face_detection()
    return jsonify({
        'success': True,
        'enabled': enabled
    })


@app.route('/toggle_face_zoom', methods=['POST'])
def toggle_face_zoom():
    enabled = transformer.toggle_face_zoom()
    return jsonify({
        'success': True,
        'enabled': enabled
    })


@app.route('/toggle_drawing_mode', methods=['POST'])
def toggle_drawing_mode():
    enabled = transformer.toggle_drawing_mode()
    return jsonify({
        'success': True,
        'enabled': enabled
    })


@app.route('/set_brush_color', methods=['POST'])
def set_brush_color():
    data = request.get_json()
    color = data.get('color', [255, 255, 255])
    transformer.set_brush_color(tuple(color))
    return jsonify({'success': True})


@app.route('/set_brush_size', methods=['POST'])
def set_brush_size():
    data = request.get_json()
    size = data.get('size', 5)
    transformer.set_brush_size(size)
    return jsonify({'success': True})


@app.route('/draw', methods=['POST'])
def draw():
    data = request.get_json()
    start_pos = tuple(data.get('start_pos', [0, 0]))
    end_pos = tuple(data.get('end_pos', [0, 0]))

    processed_image = transformer.draw_on_image(start_pos, end_pos)
    processed_base64 = image_to_base64(processed_image)

    return jsonify({
        'success': True,
        'processed_image': processed_base64
    })


@app.route('/preview', methods=['POST'])
def preview_operation():
    try:
        data = request.get_json()
        image_data = data['image']
        operation = data['operation']
        parameters = data.get('parameters', {})

        # Convert base64 to OpenCV image
        if image_data.startswith('data:image'):
            image = base64_to_image(image_data.split(',')[1])
        else:
            image = base64_to_image(image_data)

        # Set image if not set
        if transformer.current_image is None:
            transformer.set_image(image)

        # Get preview based on operation type
        preview_image = None

        if operation == 'preview_shape':
            shape_type = parameters.get('shape_type', 'rectangle')
            coordinates = parameters.get('coordinates', [[0, 0], [100, 100]])
            color = parameters.get('color', [255, 255, 255])
            thickness = parameters.get('thickness', 2)
            preview_image = transformer.get_preview_with_shape(shape_type, coordinates, color, thickness)

        elif operation == 'preview_crop':
            x1 = parameters.get('x1', 0)
            y1 = parameters.get('y1', 0)
            x2 = parameters.get('x2', 100)
            y2 = parameters.get('y2', 100)
            preview_image = transformer.get_preview_with_crop(x1, y1, x2, y2)

        elif operation == 'preview_rotate':
            angle = parameters.get('angle', 0)
            preview_image = transformer.get_preview_with_rotation(angle)

        else:
            preview_image = transformer.get_current_image()

        if preview_image is not None:
            preview_base64 = image_to_base64(preview_image)
            return jsonify({
                'success': True,
                'preview_image': preview_base64
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Preview not available'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/chain_operations', methods=['POST'])
def chain_operations():
    try:
        data = request.get_json()
        image_data = data['image']
        operations = data['operations']

        # Convert base64 to OpenCV image
        if image_data.startswith('data:image'):
            image = base64_to_image(image_data.split(',')[1])
        else:
            image = base64_to_image(image_data)

        # Set initial image
        transformer.set_image(image)

        # Apply all operations in chain
        for op in operations:
            apply_image_operations(op['operation'], op.get('parameters', {}))

        # Get final image
        processed_image = transformer.get_current_image()

        # Convert back to base64
        processed_base64 = image_to_base64(processed_image)

        return jsonify({
            'success': True,
            'processed_image': processed_base64
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def apply_image_operations(operation, parameters):
    """Apply various image processing operations"""

    if operation == 'original':
        return transformer.reset()

    elif operation == 'reset_chain':
        return transformer.reset()

    # Geometric operations
    elif operation == 'translate':
        x = parameters.get('x', 0)
        y = parameters.get('y', 0)
        return transformer.translate(x, y)

    elif operation == 'rotate':
        angle = parameters.get('value', 45)
        return transformer.rotate(angle)

    elif operation == 'resize':
        width = parameters.get('width')
        height = parameters.get('height')
        return transformer.resize(width, height)

    elif operation == 'flip_h':
        return transformer.flip(1)

    elif operation == 'flip_v':
        return transformer.flip(0)

    elif operation == 'flip_both':
        return transformer.flip(-1)

    elif operation == 'crop':
        x1 = parameters.get('x1', 0)
        y1 = parameters.get('y1', 0)
        x2 = parameters.get('x2', 100)
        y2 = parameters.get('y2', 100)
        return transformer.crop(y1, y2, x1, x2)

    # Color operations
    elif operation == 'grayscale':
        return transformer.color_operations('grayscale')

    elif operation == 'hsv':
        return transformer.color_operations('hsv')

    elif operation == 'lab':
        return transformer.color_operations('lab')

    elif operation == 'negative':
        return transformer.color_operations('negative')

    elif operation == 'sepia':
        return transformer.color_operations('sepia')

    elif operation == 'warm':
        return transformer.color_operations('warm')

    elif operation == 'cool':
        return transformer.color_operations('cool')

    elif operation == 'brightness':
        value = int(parameters.get('value', 0))
        return transformer.color_operations('adjust_brightness', {'value': value})

    elif operation == 'contrast':
        value = int(parameters.get('value', 0))
        return transformer.color_operations('adjust_contrast', {'value': value})

    elif operation == 'saturation':
        value = int(parameters.get('value', 0))
        return transformer.color_operations('adjust_saturation', {'value': value})

    elif operation == 'hue':
        value = int(parameters.get('value', 0))
        return transformer.color_operations('adjust_hue', {'value': value})

    elif operation == 'color_overlay':
        color = parameters.get('color', [255, 0, 0])
        alpha = parameters.get('alpha', 0.5)
        return transformer.color_operations('color_overlay', {'color': color, 'alpha': alpha})

    elif operation == 'histogram_equalization':
        return transformer.histogram_equalization()

    # Filter operations
    elif operation == 'blur':
        kernel_size = parameters.get('kernel_size', 15)
        return transformer.filter_operations('blur', {'kernel_size': kernel_size})

    elif operation == 'sharpen':
        return transformer.filter_operations('sharpen')

    elif operation == 'emboss':
        return transformer.filter_operations('emboss')

    elif operation == 'gaussian_blur':
        kernel_size = parameters.get('kernel_size', 15)
        return transformer.filter_operations('gaussian_blur', {'kernel_size': kernel_size})

    elif operation == 'median_blur':
        kernel_size = parameters.get('kernel_size', 15)
        return transformer.filter_operations('median_blur', {'kernel_size': kernel_size})

    elif operation == 'bilateral_filter':
        return transformer.filter_operations('bilateral_filter')

    # Edge detection operations
    elif operation.startswith('edge_'):
        method = operation.split('_')[1]
        threshold1 = parameters.get('threshold1', 100)
        threshold2 = parameters.get('threshold2', 200)
        return transformer.edge_detection(method, {'threshold1': threshold1, 'threshold2': threshold2})

    # Morphological operations
    elif operation == 'erode':
        kernel_size = parameters.get('kernel_size', 3)
        return transformer.morphological_operations('erode', kernel_size)

    elif operation == 'dilate':
        kernel_size = parameters.get('kernel_size', 3)
        return transformer.morphological_operations('dilate', kernel_size)

    elif operation == 'open':
        kernel_size = parameters.get('kernel_size', 3)
        return transformer.morphological_operations('open', kernel_size)

    elif operation == 'close':
        kernel_size = parameters.get('kernel_size', 3)
        return transformer.morphological_operations('close', kernel_size)

    elif operation == 'gradient':
        kernel_size = parameters.get('kernel_size', 3)
        return transformer.morphological_operations('gradient', kernel_size)

    # Threshold operations
    elif operation.startswith('threshold_'):
        method = operation.split('_')[1]
        threshold = parameters.get('threshold', 127)
        max_value = parameters.get('max_value', 255)
        return transformer.threshold_operations(method, threshold, max_value)

    # Arithmetic operations
    elif operation == 'arithmetic_add':
        value = int(parameters.get('value', 10))
        return transformer.arithmetic_operations('add', value)

    elif operation == 'arithmetic_subtract':
        value = int(parameters.get('value', 10))
        return transformer.arithmetic_operations('subtract', value)

    elif operation == 'arithmetic_multiply':
        value = int(parameters.get('value', 2))
        return transformer.arithmetic_operations('multiply', value)

    elif operation == 'arithmetic_divide':
        value = int(parameters.get('value', 2))
        return transformer.arithmetic_operations('divide', value)

    # Shape operations
    elif operation == 'draw_rectangle':
        x1 = parameters.get('x1', 50)
        y1 = parameters.get('y1', 50)
        x2 = parameters.get('x2', 200)
        y2 = parameters.get('y2', 200)
        color = parameters.get('color', [255, 255, 255])
        thickness = parameters.get('thickness', 2)
        return transformer.create_shapes('rectangle', [(x1, y1), (x2, y2)], color, thickness)

    elif operation == 'draw_circle':
        center = parameters.get('center', [150, 150])
        radius = parameters.get('radius', 50)
        color = parameters.get('color', [255, 255, 255])
        thickness = parameters.get('thickness', 2)
        return transformer.create_shapes('circle', [center, radius], color, thickness)

    elif operation == 'draw_line':
        start = parameters.get('start', [50, 50])
        end = parameters.get('end', [200, 200])
        color = parameters.get('color', [255, 255, 255])
        thickness = parameters.get('thickness', 2)
        return transformer.create_shapes('line', [start, end], color, thickness)

    elif operation == 'draw_polygon':
        points = parameters.get('points', [[100, 100], [200, 50], [300, 100], [250, 200], [150, 200]])
        color = parameters.get('color', [255, 255, 255])
        thickness = parameters.get('thickness', -1)
        return transformer.create_shapes('polygon', points, color, thickness)

    elif operation == 'draw_text':
        text = parameters.get('text', 'Hello')
        position = parameters.get('position', [50, 50])
        font_scale = parameters.get('font_scale', 1.0)
        color = parameters.get('color', [255, 255, 255])
        thickness = parameters.get('thickness', 2)
        return transformer.create_shapes('text', [text, position, font_scale], color, thickness)

    # Bitwise operations
    elif operation == 'bitwise_and':
        shape_type = parameters.get('shape_type', 'rectangle')
        coordinates = parameters.get('coordinates', [[50, 50], [200, 200]])
        return transformer.bitwise_operations('and', shape_type, coordinates)

    elif operation == 'bitwise_or':
        shape_type = parameters.get('shape_type', 'rectangle')
        coordinates = parameters.get('coordinates', [[50, 50], [200, 200]])
        return transformer.bitwise_operations('or', shape_type, coordinates)

    elif operation == 'bitwise_xor':
        shape_type = parameters.get('shape_type', 'rectangle')
        coordinates = parameters.get('coordinates', [[50, 50], [200, 200]])
        return transformer.bitwise_operations('xor', shape_type, coordinates)

    elif operation == 'bitwise_not':
        return transformer.bitwise_operations('not')

    # Noise operations
    elif operation == 'gaussian_noise':
        mean = parameters.get('mean', 0)
        std = parameters.get('std', 25)
        return transformer.noise_operations('gaussian_noise', {'mean': mean, 'std': std})

    elif operation == 'salt_pepper_noise':
        amount = parameters.get('amount', 0.05)
        s_vs_p = parameters.get('s_vs_p', 0.5)
        return transformer.noise_operations('salt_pepper_noise', {'amount': amount, 's_vs_p': s_vs_p})

    elif operation == 'poisson_noise':
        return transformer.noise_operations('poisson_noise')

    elif operation == 'speckle_noise':
        return transformer.noise_operations('speckle_noise')

    # AI operations
    elif operation == 'detect_faces':
        return transformer.detect_faces()

    else:
        return transformer.get_current_image()


if __name__ == '__main__':
    print("🎯 AW Shots запущен на http://localhost:5000")
    print("📷 Камера будет работать только на localhost!")
    app.run(debug=True, host='127.0.0.1', port=5000)