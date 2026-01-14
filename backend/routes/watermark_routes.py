from flask import Blueprint, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from backend.algorithms.watermark_embedding import WatermarkEmbedder
from backend.algorithms.watermark_extraction import WatermarkExtractor

watermark_bp = Blueprint('watermark', __name__, url_prefix='/api/watermark')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@watermark_bp.route('/embed', methods=['POST'])
def embed_watermark():
    """Handle watermark embedding request"""
    try:
        if 'cover' not in request.files or 'watermark' not in request.files:
            return jsonify({'error': 'Missing required files'}), 400
            
        cover_file = request.files['cover']
        watermark_file = request.files['watermark']
        
        if not (cover_file and watermark_file and 
                allowed_file(cover_file.filename) and 
                allowed_file(watermark_file.filename)):
            return jsonify({'error': 'Invalid file format'}), 400

        # Read images
        cover_img = cv2.imdecode(
            np.frombuffer(cover_file.read(), np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        watermark_img = cv2.imdecode(
            np.frombuffer(watermark_file.read(), np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        
        # Initialize embedder
        embedder = WatermarkEmbedder()
        
        # Perform embedding
        result = embedder.embed(cover_img, watermark_img)
        
        # Save watermarked image
        output_filename = f"watermarked_{uuid.uuid4()}.png"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        cv2.imwrite(output_path, result['watermarked_image'])
        
        return jsonify({
            'success': True,
            'filename': output_filename,
            'metrics': result['metrics'],
            'alpha': float(result['alpha'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@watermark_bp.route('/extract', methods=['POST'])
def extract_watermark():
    """Handle watermark extraction request"""
    try:
        if 'watermarked' not in request.files or 'original' not in request.files:
            return jsonify({'error': 'Missing required files'}), 400
            
        watermarked_file = request.files['watermarked']
        original_file = request.files['original']
        alpha = float(request.form['alpha'])
        
        # Read images
        watermarked_img = cv2.imdecode(
            np.frombuffer(watermarked_file.read(), np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        original_img = cv2.imdecode(
            np.frombuffer(original_file.read(), np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
        
        # Initialize extractor
        extractor = WatermarkExtractor()
        
        # Perform extraction
        result = extractor.extract(watermarked_img, original_img, alpha)
        
        if not result['extraction_successful']:
            return jsonify({'error': 'Extraction failed'}), 400
            
        # Save extracted watermark
        output_filename = f"extracted_{uuid.uuid4()}.png"
        output_path = os.path.join(UPLOAD_FOLDER, output_filename)
        cv2.imwrite(output_path, result['extracted_watermark'])
        
        return jsonify({
            'success': True,
            'filename': output_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500