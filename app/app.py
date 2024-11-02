from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model globally
model = YOLO('yolov8n.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_coco_classes():
    """Get COCO classes dictionary"""
    return model.names

def detect_objects(image_path, selected_classes, confidence_threshold=0.3):
    """Detect selected objects in the image"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")

    results = model(image)[0]
    detections = {class_name: 0 for class_name in selected_classes}
    
    # Process detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)
        class_name = model.names[class_id]
        
        if class_name in selected_classes and score > confidence_threshold:
            detections[class_name] += 1
            
            # Draw bounding box
            cv2.rectangle(image, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 
                         2)
            
            # Add label
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image, 
                       label, 
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       2)

    # Convert image to base64 for display
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return image_base64, detections

@app.route('/')
def index():
    classes = get_coco_classes()
    return render_template('index.html', classes=classes)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    selected_classes = request.form.getlist('classes[]')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not selected_classes:
        return jsonify({'error': 'No classes selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            image_base64, detections = detect_objects(filepath, selected_classes)
            os.remove(filepath)  # Clean up uploaded file
            
            return jsonify({
                'success': True,
                'detections': detections,
                'image': f'data:image/jpeg;base64,{image_base64}'
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)  # Clean up uploaded file
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=True)