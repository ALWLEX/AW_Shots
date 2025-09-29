let currentImage = null;
let originalImage = null;
let operationChain = [];
let history = [];
let historyIndex = -1;
let classificationEnabled = true;
let currentLanguage = 'en';
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let isDragging = false;
let currentShape = null;
let dragStart = null;
let currentTool = 'select';
let faceDetectionEnabled = false;
let faceZoomEnabled = false;

let imageStates = {
    current: null,
    history: [],
    historyIndex: -1
};

let cameraStream = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeAdvancedColorPicker();
    initializeDrawingCanvas();
});

function initializeEventListeners() {
    document.getElementById('fileInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            loadImage(file);
        }
    });

    document.getElementById('brightness').addEventListener('input', function() {
        document.getElementById('brightnessValue').textContent = this.value;
        if (currentImage) {
            debouncedAdjustment('brightness', parseInt(this.value));
        }
    });

    document.getElementById('contrast').addEventListener('input', function() {
        document.getElementById('contrastValue').textContent = this.value;
        if (currentImage) {
            debouncedAdjustment('contrast', parseInt(this.value));
        }
    });

    document.getElementById('saturation').addEventListener('input', function() {
        document.getElementById('saturationValue').textContent = this.value;
        if (currentImage) {
            debouncedAdjustment('saturation', parseInt(this.value));
        }
    });

    document.getElementById('threshold1').addEventListener('input', function() {
        document.getElementById('threshold1Value').textContent = this.value;
    });

    document.getElementById('threshold2').addEventListener('input', function() {
        document.getElementById('threshold2Value').textContent = this.value;
    });

    document.getElementById('colorHex').addEventListener('input', function() {
        const color = this.value;
        if (isValidHex(color)) {
            document.getElementById('colorDisplay').style.background = color;
            updateBrushPreview();
        }
    });

    document.getElementById('brushSize').addEventListener('input', function() {
        updateBrushPreview();
    });

    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.borderColor = '#7289da';
        this.style.background = 'rgba(114, 137, 218, 0.1)';
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.borderColor = '#40444b';
        this.style.background = '';
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.borderColor = '#40444b';
        this.style.background = '';

        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            loadImage(file);
        } else {
            alert(getTranslation('Please drop a valid image file'));
        }
    });

    const processedImage = document.getElementById('processedImage');
    processedImage.addEventListener('mousedown', startDrawing);
    processedImage.addEventListener('mousemove', draw);
    processedImage.addEventListener('mouseup', stopDrawing);
    processedImage.addEventListener('mouseleave', stopDrawing);

    processedImage.addEventListener('touchstart', handleTouchStart);
    processedImage.addEventListener('touchmove', handleTouchMove);
    processedImage.addEventListener('touchend', stopDrawing);
}

function initializeAdvancedColorPicker() {
    const rainbow = document.getElementById('rainbowGradient');
    if (rainbow) {
        rainbow.style.background = 'linear-gradient(to right, red, orange, yellow, green, blue, indigo, violet)';
        rainbow.addEventListener('click', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const hue = x / rect.width;
            const rgb = hsvToRgb(hue, 1, 1);
            const hex = rgbToHex(rgb[0], rgb[1], rgb[2]);
            document.getElementById('colorHex').value = hex;
            document.getElementById('colorDisplay').style.background = hex;
            updateBrushPreview();
        });
    }

    const colorWheel = document.getElementById('colorWheel');
    if (colorWheel) {
        colorWheel.style.background = 'conic-gradient(red, yellow, lime, aqua, blue, magenta, red)';
        colorWheel.addEventListener('click', function(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const dx = x - centerX;
            const dy = y - centerY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const maxDistance = centerX;

            if (distance <= maxDistance) {
                const angle = Math.atan2(dy, dx);
                const hue = (angle + Math.PI) / (2 * Math.PI);
                const saturation = distance / maxDistance;
                const rgb = hsvToRgb(hue, saturation, 1);
                const hex = rgbToHex(rgb[0], rgb[1], rgb[2]);
                document.getElementById('colorHex').value = hex;
                document.getElementById('colorDisplay').style.background = hex;
                updateBrushPreview();
            }
        });
    }
}

function initializeDrawingCanvas() {
    const canvas = document.getElementById('drawingCanvas');
    const processedImg = document.getElementById('processedImage');

    if (canvas && processedImg) {
        const updateCanvasSize = () => {
            if (processedImg.offsetWidth > 0 && processedImg.offsetHeight > 0) {
                canvas.width = processedImg.offsetWidth;
                canvas.height = processedImg.offsetHeight;

                const ctx = canvas.getContext('2d');
                ctx.lineJoin = 'round';
                ctx.lineCap = 'round';
                ctx.globalCompositeOperation = 'source-over';

                console.log(`Canvas initialized: ${canvas.width}x${canvas.height}`);
            }
        };

        processedImg.onload = updateCanvasSize;

        window.addEventListener('resize', updateCanvasSize);

        if (processedImg.complete && processedImg.naturalWidth > 0) {
            updateCanvasSize();
        }

        const setupEventListeners = () => {
            canvas.removeEventListener('mousedown', startDrawing);
            canvas.removeEventListener('mousemove', draw);
            canvas.removeEventListener('mouseup', stopDrawing);
            canvas.removeEventListener('mouseout', stopDrawing);
            canvas.removeEventListener('touchstart', startDrawing);
            canvas.removeEventListener('touchmove', draw);
            canvas.removeEventListener('touchend', stopDrawing);

            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);

            canvas.addEventListener('touchstart', startDrawing);
            canvas.addEventListener('touchmove', draw);
            canvas.addEventListener('touchend', stopDrawing);
        };

        setupEventListeners();
    }
}

let adjustmentTimeout;
function debouncedAdjustment(type, value) {
    clearTimeout(adjustmentTimeout);
    adjustmentTimeout = setTimeout(() => {
        applyAdjustment(type, value);
    }, 100);
}

function switchLanguage(lang) {
    currentLanguage = lang;

    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    document.querySelectorAll('[data-lang]').forEach(element => {
        if (element.getAttribute('data-lang') === lang) {
            element.style.display = '';
        } else {
            element.style.display = 'none';
        }
    });
}

function getTranslation(key) {
    const translations = {
        'en': {
            'Please drop a valid image file': 'Please drop a valid image file',
            'Error processing image': 'Error processing image',
            'No operations in chain': 'No operations in chain',
            'Detected': 'Detected',
            'Confidence': 'Confidence'
        },
        'ru': {
            'Please drop a valid image file': 'Пожалуйста, перетащите валидный файл изображения',
            'Error processing image': 'Ошибка обработки изображения',
            'No operations in chain': 'Нет операций в цепочке',
            'Detected': 'Обнаружено',
            'Confidence': 'Уверенность'
        }
    };
    return translations[currentLanguage][key] || key;
}

function loadImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        originalImage = e.target.result;
        currentImage = e.target.result;

        imageStates.history = [currentImage];
        imageStates.historyIndex = 0;
        updateHistoryButtons();

        displayImages();
        processImage(currentImage, 'original');
        showMainInterface();
    };
    reader.readAsDataURL(file);
}

function displayImages() {
    document.getElementById('originalImage').src = originalImage;
    document.getElementById('processedImage').src = currentImage;

    const processedImg = document.getElementById('processedImage');
    const canvas = document.getElementById('drawingCanvas');

    processedImg.onload = function() {
        canvas.width = processedImg.offsetWidth;
        canvas.height = processedImg.offsetHeight;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    };

    if (processedImg.complete) {
        processedImg.onload();
    }
}

function showMainInterface() {
    document.getElementById('uploadArea').style.display = 'none';
    document.getElementById('imageComparison').style.display = 'grid';
    document.getElementById('predictionResult').style.display = 'block';
    document.getElementById('historySection').style.display = 'block';
    updateHistoryDisplay();
}

function saveToHistory() {
    imageStates.history = imageStates.history.slice(0, imageStates.historyIndex + 1);
    imageStates.history.push(currentImage);
    imageStates.historyIndex++;
    updateHistoryButtons();
    updateHistoryDisplay();
}

function updateHistoryButtons() {
    document.getElementById('undoBtn').disabled = imageStates.historyIndex <= 0;
    document.getElementById('redoBtn').disabled = imageStates.historyIndex >= imageStates.history.length - 1;
}

function updateHistoryDisplay() {
    const historyItems = document.getElementById('historyItems');
    historyItems.innerHTML = '';

    imageStates.history.forEach((image, index) => {
        const item = document.createElement('div');
        item.className = `history-item ${index === imageStates.historyIndex ? 'active' : ''}`;
        item.style.backgroundImage = `url(${image})`;
        item.style.backgroundSize = 'cover';
        item.onclick = () => loadHistoryState(index);
        historyItems.appendChild(item);
    });
}

function loadHistoryState(index) {
    imageStates.historyIndex = index;
    currentImage = imageStates.history[index];
    document.getElementById('processedImage').src = currentImage;
    updateHistoryButtons();
    updateHistoryDisplay();
}

function undo() {
    if (imageStates.historyIndex > 0) {
        imageStates.historyIndex--;
        currentImage = imageStates.history[imageStates.historyIndex];
        document.getElementById('processedImage').src = currentImage;
        updateHistoryButtons();
        updateHistoryDisplay();
    }
}

function redo() {
    if (imageStates.historyIndex < imageStates.history.length - 1) {
        imageStates.historyIndex++;
        currentImage = imageStates.history[imageStates.historyIndex];
        document.getElementById('processedImage').src = currentImage;
        updateHistoryButtons();
        updateHistoryDisplay();
    }
}

async function checkCameraSupport() {
    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.error('getUserMedia not supported');
            return false;
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        console.log('Available cameras:', videoDevices);
        return videoDevices.length > 0;

    } catch (error) {
        console.error('Camera check error:', error);
        return false;
    }
}



function startWebcam() {
    console.log('AW Shots: Starting webcam...');

    const video = document.getElementById('webcam');
    const webcamContainer = document.getElementById('webcamContainer');
    const uploadArea = document.getElementById('uploadArea');

    if (navigator.mediaDevices.getUserMedia) {
        console.log('getUserMedia supported');

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                console.log('Camera access granted');
                cameraStream = stream;
                video.srcObject = stream;

                webcamContainer.style.display = 'block';
                uploadArea.style.display = 'none';

                console.log('Webcam started successfully');
            })
            .catch(function(error) {
                console.log('Camera error:', error);
                handleCameraError(error);
            });
    } else {
        console.log('getUserMedia not supported');
        alert('Ваш браузер не поддерживает доступ к камере.');
    }
}

function handleCameraError(error) {
    let errorMessage = 'Ошибка доступа к камере: ';

    if (error.name === 'NotAllowedError') {
        errorMessage = '❌ Вы запретили доступ к камере.\n\nЧтобы разрешить:\n1. Нажмите на значок камеры в адресной строке\n2. Выберите "Разрешить"\n3. Обновите страницу';
    } else if (error.name === 'NotFoundError') {
        errorMessage = '📷 Камера не найдена.\n\nУбедитесь, что камера подключена.';
    } else if (error.name === 'NotSupportedError') {
        errorMessage = '🌐 Ваш браузер не поддерживает камеру.\n\nПопробуйте Chrome или Firefox.';
    } else {
        errorMessage = '❌ Ошибка: ' + error.message;
    }

    alert(errorMessage);
}

function stopWebcam() {
    console.log('AW Shots: Stopping webcam...');

    const video = document.getElementById('webcam');
    const webcamContainer = document.getElementById('webcamContainer');
    const uploadArea = document.getElementById('uploadArea');

    if (cameraStream) {
        var tracks = cameraStream.getTracks();
        for (var i = 0; i < tracks.length; i++) {
            var track = tracks[i];
            track.stop();
        }
        cameraStream = null;
    }

    if (video.srcObject) {
        video.srcObject = null;
    }

    webcamContainer.style.display = 'none';
    uploadArea.style.display = 'block';
}

function captureImage() {
    const video = document.getElementById('webcam');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL('image/png');

    originalImage = imageData;
    currentImage = imageData;

    imageStates.history = [currentImage];
    imageStates.historyIndex = 0;

    displayImages();
    showMainInterface();

    stopWebcam();

    console.log('AW Shots: Image captured from webcam');
}


async function processImage(imageData, operation, parameters = {}) {
    showLoading(true);
    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: imageData,
                operation: operation,
                parameters: parameters
            })
        });

        const result = await response.json();

        if (result.success) {
            currentImage = 'data:image/jpeg;base64,' + result.processed_image;
            document.getElementById('processedImage').src = currentImage;

            saveToHistory();

            if (result.prediction && classificationEnabled) {
                displayPrediction(result.prediction);
            }
        } else {
            alert(getTranslation('Error processing image') + ': ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert(getTranslation('Error processing image'));
    } finally {
        showLoading(false);
    }
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = show ? 'block' : 'none';
    }
}

function displayPrediction(prediction) {
    const resultDiv = document.getElementById('predictionContent');
    const confidencePercent = (prediction.confidence * 100).toFixed(2);

    resultDiv.innerHTML = `
        <h4>${getTranslation('Detected')}: ${prediction.class_name}</h4>
        <p>${getTranslation('Confidence')}: ${confidencePercent}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
        </div>
    `;
}

function startDrawing(e) {
    if (currentTool !== 'brush') return;

    isDrawing = true;
    const canvas = document.getElementById('drawingCanvas');
    const rect = canvas.getBoundingClientRect();

    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);

    lastX = clientX - rect.left;
    lastY = clientY - rect.top;

    e.preventDefault();
}

function draw(e) {
    if (!isDrawing) return;

    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);

    const currentX = clientX - rect.left;
    const currentY = clientY - rect.top;

    const color = document.getElementById('colorHex').value;
    const rgb = hexToRgb(color);
    const brushSize = parseInt(document.getElementById('brushSize').value);

    ctx.strokeStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
    ctx.lineWidth = brushSize;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();

    lastX = currentX;
    lastY = currentY;

    e.preventDefault();
}


function stopDrawing() {
    if (!isDrawing) return;

    isDrawing = false;

    setTimeout(saveDrawingToImage, 100);
}

function saveDrawingToImage() {
    const canvas = document.getElementById('drawingCanvas');
    const processedImg = document.getElementById('processedImage');

    if (processedImg.src && processedImg.src !== '') {
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');

        tempCanvas.width = processedImg.naturalWidth || processedImg.width;
        tempCanvas.height = processedImg.naturalHeight || processedImg.height;

        tempCtx.drawImage(processedImg, 0, 0, tempCanvas.width, tempCanvas.height);

        const scaleX = tempCanvas.width / canvas.width;
        const scaleY = tempCanvas.height / canvas.height;

        tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, tempCanvas.width, tempCanvas.height);

        currentImage = tempCanvas.toDataURL('image/png');
        processedImg.src = currentImage;

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        saveToHistory();
    }
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    startDrawing(mouseEvent);
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    draw(mouseEvent);
}

function startShapeDrag(e) {
    if (currentTool !== 'select') return;

    isDragging = true;
    dragStart = { x: e.clientX, y: e.clientY };

    const rect = e.target.getBoundingClientRect();
    currentShape = {
        type: 'rectangle',
        x: rect.left,
        y: rect.top,
        width: rect.width,
        height: rect.height
    };
}

function applyFilter(filter) {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }
    processImage(currentImage, filter);
    addToChain(filter);
}

function applyTransformation(transformation) {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }
    processImage(currentImage, transformation);
    addToChain(transformation);
}

function applyAdjustment(type, value) {
    if (!currentImage) return;
    processImage(currentImage, type, { value: value });
}

function applyEdgeDetection(type) {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }
    const threshold1 = parseInt(document.getElementById('threshold1').value);
    const threshold2 = parseInt(document.getElementById('threshold2').value);

    processImage(currentImage, `edge_${type}`, {
        threshold1: threshold1,
        threshold2: threshold2
    });
    addToChain(`edge_${type}`, { threshold1, threshold2 });
    closeModal('edgeDetectionModal');
}

function showRealTimePreview(operation, parameters) {
    if (!currentImage) return;

    const preview = document.getElementById('previewOverlay');
    preview.innerHTML = '';

    if (operation === 'shape') {
        const shape = document.createElement('div');
        shape.className = 'preview-shape';
        shape.style.left = parameters.x + 'px';
        shape.style.top = parameters.y + 'px';
        shape.style.width = parameters.width + 'px';
        shape.style.height = parameters.height + 'px';
        preview.appendChild(shape);
    } else if (operation === 'crop') {
        const crop = document.createElement('div');
        crop.className = 'preview-crop';
        crop.style.left = parameters.x1 + 'px';
        crop.style.top = parameters.y1 + 'px';
        crop.style.width = (parameters.x2 - parameters.x1) + 'px';
        crop.style.height = (parameters.y2 - parameters.y1) + 'px';
        preview.appendChild(crop);
    }
}

function showTransformModal(type) {
    const modal = document.getElementById('transformModal');
    const content = document.getElementById('transformContent');

    let html = '';
    if (type === 'rotate') {
        html = `
            <div class="slider-group">
                <div class="slider-label">
                    <span data-lang="en">Angle</span>
                    <span data-lang="ru" style="display: none;">Угол</span>
                    <span id="angleValue">0</span>°
                </div>
                <input type="range" class="slider" id="angleSlider" min="-180" max="180" value="0">
            </div>
            <div class="toolbar-btn" onclick="applyRotation()" style="width: 100%; margin-top: 15px; text-align: center;">
                <span data-lang="en">Apply Rotation</span>
                <span data-lang="ru" style="display: none;">Применить поворот</span>
            </div>
        `;
    } else if (type === 'resize') {
        html = `
            <div class="coordinate-inputs">
                <input type="number" class="coordinate-input" id="resizeWidth" placeholder="Width" value="800">
                <input type="number" class="coordinate-input" id="resizeHeight" placeholder="Height" value="600">
            </div>
            <div class="toolbar-btn" onclick="applyResize()" style="width: 100%; margin-top: 15px; text-align: center;">
                <span data-lang="en">Resize</span>
                <span data-lang="ru" style="display: none;">Изменить размер</span>
            </div>
        `;
    } else if (type === 'crop') {
        html = `
            <div class="coordinate-inputs">
                <div class="coordinate-row">
                    <input type="number" class="coordinate-input" id="cropX1" placeholder="X1" value="0">
                    <input type="number" class="coordinate-input" id="cropY1" placeholder="Y1" value="0">
                </div>
                <div class="coordinate-row">
                    <input type="number" class="coordinate-input" id="cropX2" placeholder="X2" value="200">
                    <input type="number" class="coordinate-input" id="cropY2" placeholder="Y2" value="200">
                </div>
            </div>
            <div class="toolbar-btn" onclick="applyCrop()" style="width: 100%; margin-top: 15px; text-align: center;">
                <span data-lang="en">Apply Crop</span>
                <span data-lang="ru" style="display: none;">Применить обрезку</span>
            </div>
        `;
    }

    content.innerHTML = html;

    if (type === 'rotate') {
        document.getElementById('angleSlider').addEventListener('input', function() {
            document.getElementById('angleValue').textContent = this.value;
            previewRotation(parseInt(this.value));
        });
    }

    modal.style.display = 'flex';
}

function showDrawingModal(shape) {
    const modal = document.getElementById('drawingModal');
    const content = document.getElementById('drawingContent');

    let html = '';
    if (shape === 'rectangle') {
        html = `
            <div class="coordinate-inputs">
                <div class="coordinate-row">
                    <input type="number" class="coordinate-input" id="rectX1" placeholder="X1" value="50">
                    <input type="number" class="coordinate-input" id="rectY1" placeholder="Y1" value="50">
                </div>
                <div class="coordinate-row">
                    <input type="number" class="coordinate-input" id="rectX2" placeholder="X2" value="200">
                    <input type="number" class="coordinate-input" id="rectY2" placeholder="Y2" value="200">
                </div>
            </div>
            <div class="toolbar-btn" onclick="drawRectangle()" style="width: 100%; margin-top: 15px; text-align: center;">
                <span data-lang="en">Draw Rectangle</span>
                <span data-lang="ru" style="display: none;">Нарисовать прямоугольник</span>
            </div>
        `;
    } else if (shape === 'circle') {
        html = `
            <div class="coordinate-inputs">
                <input type="number" class="coordinate-input" id="circleX" placeholder="Center X" value="150">
                <input type="number" class="coordinate-input" id="circleY" placeholder="Center Y" value="150">
                <input type="number" class="coordinate-input" id="circleRadius" placeholder="Radius" value="50">
            </div>
            <div class="toolbar-btn" onclick="drawCircle()" style="width: 100%; margin-top: 15px; text-align: center;">
                <span data-lang="en">Draw Circle</span>
                <span data-lang="ru" style="display: none;">Нарисовать круг</span>
            </div>
        `;
    }

    content.innerHTML = html;
    modal.style.display = 'flex';
}

function showEdgeDetectionModal() {
    document.getElementById('edgeDetectionModal').style.display = 'flex';
}

function showAdvancedColorPicker() {
    document.getElementById('advancedColorPicker').style.display = 'flex';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function applyRotation() {
    const angle = parseInt(document.getElementById('angleSlider').value);
    processImage(currentImage, 'rotate', { value: angle });
    addToChain('rotate', { value: angle });
    closeModal('transformModal');
}

function applyResize() {
    const width = parseInt(document.getElementById('resizeWidth').value);
    const height = parseInt(document.getElementById('resizeHeight').value);
    processImage(currentImage, 'resize', { width: width, height: height });
    addToChain('resize', { width, height });
    closeModal('transformModal');
}

function applyCrop() {
    const x1 = parseInt(document.getElementById('cropX1').value);
    const y1 = parseInt(document.getElementById('cropY1').value);
    const x2 = parseInt(document.getElementById('cropX2').value);
    const y2 = parseInt(document.getElementById('cropY2').value);
    processImage(currentImage, 'crop', { x1: x1, y1: y1, x2: x2, y2: y2 });
    addToChain('crop', { x1, y1, x2, y2 });
    closeModal('transformModal');
}

function drawRectangle() {
    const x1 = parseInt(document.getElementById('rectX1').value);
    const y1 = parseInt(document.getElementById('rectY1').value);
    const x2 = parseInt(document.getElementById('rectX2').value);
    const y2 = parseInt(document.getElementById('rectY2').value);
    const color = document.getElementById('colorHex').value;
    const thickness = parseInt(document.getElementById('brushSize').value);

    const rgb = hexToRgb(color);
    processImage(currentImage, 'draw_rectangle', {
        x1: x1, y1: y1, x2: x2, y2: y2,
        color: [rgb.r, rgb.g, rgb.b],
        thickness: thickness
    });
    addToChain('draw_rectangle', { x1, y1, x2, y2, color: [rgb.r, rgb.g, rgb.b], thickness });
    closeModal('drawingModal');
}

function drawCircle() {
    const x = parseInt(document.getElementById('circleX').value);
    const y = parseInt(document.getElementById('circleY').value);
    const radius = parseInt(document.getElementById('circleRadius').value);
    const color = document.getElementById('colorHex').value;
    const thickness = parseInt(document.getElementById('brushSize').value);

    const rgb = hexToRgb(color);
    processImage(currentImage, 'draw_circle', {
        center: [x, y],
        radius: radius,
        color: [rgb.r, rgb.g, rgb.b],
        thickness: thickness
    });
    addToChain('draw_circle', { center: [x, y], radius, color: [rgb.r, rgb.g, rgb.b], thickness });
    closeModal('drawingModal');
}

function previewRotation(angle) {
    const previewElement = document.getElementById('rotationPreview');
    if (previewElement) {
        previewElement.textContent = `Rotation: ${angle}°`;
    }
}

function selectTool(tool) {
    currentTool = tool;

    document.querySelectorAll('.tool-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    const processedImage = document.getElementById('processedImage');
    if (tool === 'brush') {
        processedImage.style.cursor = 'crosshair';
    } else if (tool === 'select') {
        processedImage.style.cursor = 'move';
    } else {
        processedImage.style.cursor = 'default';
    }
}

function openFilePicker() {
    document.getElementById('fileInput').click();
}

function openUpload() {
    openFilePicker();
}

function showColorPicker() {
    document.getElementById('advancedColorPicker').style.display = 'flex';
}

function updateBrushPreview() {
    const brushPreview = document.getElementById('brushPreview');
    if (brushPreview) {
        const color = document.getElementById('colorHex').value;
        const size = document.getElementById('brushSize').value;
        brushPreview.style.background = color;
        brushPreview.style.width = size + 'px';
        brushPreview.style.height = size + 'px';
    }
}

function isValidHex(color) {
    return /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(color);
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 255, g: 255, b: 255 };
}

function rgbToHex(r, g, b) {
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

function hsvToRgb(h, s, v) {
    let r, g, b;
    let i = Math.floor(h * 6);
    let f = h * 6 - i;
    let p = v * (1 - s);
    let q = v * (1 - f * s);
    let t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function addToChain(operation, parameters = {}) {
    operationChain.push({
        operation: operation,
        parameters: parameters
    });
}

function applyChain() {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }

    if (operationChain.length === 0) {
        alert(getTranslation('No operations in chain'));
        return;
    }

    operationChain.forEach(op => {
        processImage(currentImage, op.operation, op.parameters);
    });
}

function clearChain() {
    operationChain = [];
}

function toggleClassification() {
    fetch('/toggle_classification', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        classificationEnabled = data.enabled;
        const btn = document.getElementById('classificationToggle');
        const predictionDiv = document.getElementById('predictionResult');

        if (classificationEnabled) {
            btn.innerHTML = '<i class="fas fa-robot"></i><span data-lang="en">Classification: ON</span><span data-lang="ru" style="display: none;">Классификация: ВКЛ</span>';
            btn.style.background = 'var(--success-color)';
            predictionDiv.style.display = 'block';

            if (currentImage) {
                processImage(currentImage, 'classify');
            }
        } else {
            btn.innerHTML = '<i class="fas fa-robot"></i><span data-lang="en">Classification: OFF</span><span data-lang="ru" style="display: none;">Классификация: ВЫКЛ</span>';
            btn.style.background = 'var(--danger-color)';
            predictionDiv.style.display = 'none';
        }
    })
    .catch(error => {
        console.error('Error toggling classification:', error);
    });
}

function toggleFaceDetection() {
    fetch('/toggle_face_detection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        faceDetectionEnabled = data.enabled;
        const btn = document.getElementById('faceDetectionToggle');
        if (faceDetectionEnabled) {
            btn.innerHTML = '<i class="fas fa-user"></i><span data-lang="en">Face Detection: ON</span><span data-lang="ru" style="display: none;">Детекция лиц: ВКЛ</span>';
            btn.style.background = 'var(--success-color)';
        } else {
            btn.innerHTML = '<i class="fas fa-user"></i><span data-lang="en">Face Detection: OFF</span><span data-lang="ru" style="display: none;">Детекция лиц: ВЫКЛ</span>';
            btn.style.background = 'var(--danger-color)';
        }
    });
}

function toggleFaceZoom() {
    fetch('/toggle_face_zoom', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        faceZoomEnabled = data.enabled;
        const btn = document.getElementById('faceZoomToggle');
        if (faceZoomEnabled) {
            btn.innerHTML = '<i class="fas fa-search-plus"></i><span data-lang="en">Face Zoom: ON</span><span data-lang="ru" style="display: none;">Зум лиц: ВКЛ</span>';
            btn.style.background = 'var(--success-color)';
        } else {
            btn.innerHTML = '<i class="fas fa-search-plus"></i><span data-lang="en">Face Zoom: OFF</span><span data-lang="ru" style="display: none;">Зум лиц: ВЫКЛ</span>';
            btn.style.background = 'var(--danger-color)';
        }
    });
}

function toggleDrawingMode() {
    fetch('/toggle_drawing_mode', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        const enabled = data.enabled;
        const btn = document.getElementById('drawingModeToggle');
        if (enabled) {
            btn.innerHTML = '<i class="fas fa-paint-brush"></i><span data-lang="en">Drawing: ON</span><span data-lang="ru" style="display: none;">Рисование: ВКЛ</span>';
            btn.style.background = 'var(--success-color)';
            currentTool = 'brush';
        } else {
            btn.innerHTML = '<i class="fas fa-paint-brush"></i><span data-lang="en">Drawing: OFF</span><span data-lang="ru" style="display: none;">Рисование: ВЫКЛ</span>';
            btn.style.background = 'var(--danger-color)';
            currentTool = 'select';
        }
    });
}

function detectFaces() {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }

    fetch('/detect_faces_only', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: currentImage
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentImage = 'data:image/jpeg;base64,' + data.processed_image;
            document.getElementById('processedImage').src = currentImage;
            saveToHistory();
        } else {
            alert('Error detecting faces: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error detecting faces');
    });
}

function zoomFace() {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }

    fetch('/zoom_face_only', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: currentImage
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentImage = 'data:image/jpeg;base64,' + data.processed_image;
            document.getElementById('processedImage').src = currentImage;
            saveToHistory();
        } else {
            alert('Error zooming face: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error zooming face');
    });
}

function applyColorOverlay() {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }

    const color = document.getElementById('colorHex').value;
    const rgb = hexToRgb(color);
    const alpha = 0.5; 
    processImage(currentImage, 'color_overlay', {
        color: [rgb.r, rgb.g, rgb.b],
        alpha: alpha
    });
}

function resetAll() {
    if (originalImage) {
        currentImage = originalImage;
        document.getElementById('processedImage').src = currentImage;
        processImage(originalImage, 'reset_chain');

        imageStates.history = [currentImage];
        imageStates.historyIndex = 0;
        updateHistoryButtons();
        updateHistoryDisplay();

        document.getElementById('brightness').value = 0;
        document.getElementById('contrast').value = 0;
        document.getElementById('saturation').value = 0;
        document.getElementById('brightnessValue').textContent = '0';
        document.getElementById('contrastValue').textContent = '0';
        document.getElementById('saturationValue').textContent = '0';

        const canvas = document.getElementById('drawingCanvas');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        clearChain();
    }
}

function saveImage() {
    if (!currentImage) {
        alert(getTranslation('Please upload an image first'));
        return;
    }

    const link = document.createElement('a');
    link.download = 'processed_image.jpg';
    link.href = currentImage;
    link.click();
}

updateBrushPreview();
