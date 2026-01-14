import sys
import os
# Add parent directory to path to ensure correct imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, send_file, jsonify
from flask_cors import CORS
from flask_talisman import Talisman
import cv2
import numpy as np
from backend.algorithms.watermark_embedding import WatermarkEmbedder
from backend.algorithms.watermark_extraction import WatermarkExtractor
import io
import base64
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_app():
    app = Flask(__name__)

    # Configure logging based on environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(level=getattr(logging, log_level))
    logger = logging.getLogger(__name__)

    # Security configurations
    CORS(app, resources={
        r"/embed": {"origins": os.getenv('ALLOWED_ORIGINS', '*')},
        r"/extract": {"origins": os.getenv('ALLOWED_ORIGINS', '*')}
    })

    # Security headers with Talisman
    csp = {
        'default-src': "'self'",
        'script-src': "'self' 'unsafe-inline' https://cdnjs.cloudflare.com",
        'style-src': "'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com",
        'font-src': "'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com",
        'img-src': "'self' data:",
        'connect-src': "'self'"
    }

    Talisman(app,
             content_security_policy=csp,
             force_https=False,  # Set to True in production with HTTPS
             strict_transport_security=True,
             strict_transport_security_max_age=31536000,
             session_cookie_secure=False,  # Set to True with HTTPS
             session_cookie_http_only=True,
             frame_options='DENY',
             x_content_type_options='nosniff',
             referrer_policy='strict-origin-when-cross-origin'
    )

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/embed', methods=['POST'])
    def embed():
        try:
            logger.debug("Starting embed process")
            
            if 'cover' not in request.files or 'watermark' not in request.files:
                logger.error("Missing required files")
                return jsonify({'error': 'Missing required files'}), 400

            cover_file = request.files['cover']
            watermark_file = request.files['watermark']
            
            # Log file info
            logger.debug(f"Cover file: {cover_file.filename}")
            logger.debug(f"Watermark file: {watermark_file.filename}")
            
            # Read files
            cover_bytes = cover_file.read()
            watermark_bytes = watermark_file.read()
            
            # Convert to numpy arrays with error checking
            cover_array = cv2.imdecode(np.frombuffer(cover_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            watermark_array = cv2.imdecode(np.frombuffer(watermark_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if cover_array is None:
                logger.error("Failed to decode cover image")
                return jsonify({'error': 'Invalid cover image format. Please use PNG, JPG, or JPEG.'}), 400
            if watermark_array is None:
                logger.error("Failed to decode watermark image")
                return jsonify({'error': 'Invalid watermark image format. Please use PNG, JPG, or JPEG.'}), 400
            
            # Ensure images are not empty
            if cover_array.size == 0:
                logger.error("Cover image is empty")
                return jsonify({'error': 'Cover image is empty'}), 400
            if watermark_array.size == 0:
                logger.error("Watermark image is empty")
                return jsonify({'error': 'Watermark image is empty'}), 400

            logger.debug(f"Cover shape: {cover_array.shape}")
            logger.debug(f"Watermark shape: {watermark_array.shape}")
            
            # Note: Watermark resizing will be handled by WatermarkEmbedder
            # to match the LL band dimensions after DWT decomposition
            
            # Create embedder instance
            embedder = WatermarkEmbedder()
            
            # Perform embedding with error checking
            try:
                result = embedder.embed(cover_array, watermark_array)
                if 'watermarked_image' not in result:
                    raise ValueError("Embedding failed to produce output")
                logger.debug(f"Embedding successful. Alpha: {result.get('alpha', 'N/A')}")
            except Exception as e:
                logger.error(f"Embedding failed: {str(e)}", exc_info=True)
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Return a more helpful error message
                error_msg = f'Embedding process failed: {str(e)}'
                return jsonify({'error': error_msg}), 500
            
            # Convert result to PNG and encode as base64 for JSON response
            watermarked_img = result['watermarked_image']
            
            # Ensure image is valid numpy array
            if not isinstance(watermarked_img, np.ndarray):
                raise ValueError(f"Watermarked image is not a numpy array: {type(watermarked_img)}")
            
            # Ensure image is 2D and uint8
            if watermarked_img.ndim != 2:
                raise ValueError(f"Expected 2D image, got {watermarked_img.ndim}D")
            
            # Convert to uint8 if needed
            if watermarked_img.dtype != np.uint8:
                watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
            
            # Encode to PNG
            success, img_encoded = cv2.imencode('.png', watermarked_img)
            if not success or img_encoded is None:
                raise ValueError("Failed to encode watermarked image to PNG")
            
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            
            if not img_base64:
                raise ValueError("Failed to encode image to base64")
            
            logger.debug("Successfully completed embedding and encoding")
            
            # Return JSON with image data and metrics
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'alpha': float(result.get('alpha', 0)),
                'metrics': result.get('metrics', {})
            })
            
        except Exception as e:
            logger.exception("Unexpected error during embedding")
            return jsonify({'error': str(e)}), 500

    @app.route('/extract', methods=['POST'])
    def extract():
        try:
            logger.debug("Starting extract process")
            
            if 'watermarked' not in request.files or 'original' not in request.files:
                logger.error("Missing required files")
                return jsonify({'error': 'Missing required files'}), 400
            
            if 'alpha' not in request.form:
                logger.error("Missing alpha parameter")
                return jsonify({'error': 'Missing alpha parameter'}), 400
            
            watermarked_file = request.files['watermarked']
            original_file = request.files['original']
            alpha = float(request.form['alpha'])
            
            # Read files
            watermarked_bytes = watermarked_file.read()
            original_bytes = original_file.read()
            
            # Convert to numpy arrays
            watermarked_array = cv2.imdecode(np.frombuffer(watermarked_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            original_array = cv2.imdecode(np.frombuffer(original_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if watermarked_array is None or original_array is None:
                logger.error("Failed to decode images")
                return jsonify({'error': 'Invalid image format'}), 400
            
            # Ensure images are not empty
            if watermarked_array.size == 0 or original_array.size == 0:
                logger.error("Image is empty")
                return jsonify({'error': 'Image is empty'}), 400
            
            # Ensure images have same dimensions
            if watermarked_array.shape != original_array.shape:
                logger.warning("Resizing images to match dimensions")
                watermarked_array = cv2.resize(watermarked_array, 
                                              (original_array.shape[1], original_array.shape[0]))
            
            # Create extractor instance
            extractor = WatermarkExtractor()
            
            # Perform extraction
            try:
                result = extractor.extract(watermarked_array, original_array, alpha)
                if not result.get('extraction_successful', False):
                    raise ValueError(result.get('error', 'Extraction failed'))
                logger.debug("Extraction successful")
            except Exception as e:
                logger.error(f"Extraction failed: {str(e)}", exc_info=True)
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                error_msg = f'Extraction process failed: {str(e)}'
                return jsonify({'error': error_msg}), 500
            
            # Convert extracted watermark to image
            extracted = result.get('extracted_watermark')
            if extracted is None:
                raise ValueError("No extracted watermark returned")
            
            # Ensure extracted watermark is a numpy array
            if not isinstance(extracted, np.ndarray):
                extracted = np.asarray(extracted)
            
            # Normalize and convert to uint8
            if extracted.ndim == 2:
                # Normalize to 0-255 range
                extracted = np.clip(extracted, 0, 255).astype(np.uint8)
            else:
                # If it's a 1D array (singular values), create a simple visualization
                extracted = np.clip(extracted.flatten()[:min(extracted.size, 1024)], 0, 255).astype(np.uint8)
                # Reshape to a square if possible
                size = int(np.sqrt(len(extracted)))
                if size * size == len(extracted):
                    extracted = extracted.reshape((size, size))
                else:
                    # Pad to next square
                    next_size = size + 1
                    padded = np.zeros(next_size * next_size, dtype=np.uint8)
                    padded[:len(extracted)] = extracted
                    extracted = padded.reshape((next_size, next_size))
            
            # Encode as base64
            _, img_encoded = cv2.imencode('.png', extracted)
            img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
            
            logger.debug("Successfully completed extraction")
            
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_base64}',
                'metrics': result.get('metrics', {})
            })
            
        except Exception as e:
            logger.exception("Unexpected error during extraction")
            return jsonify({'error': str(e)}), 500

    return app

if __name__ == '__main__':
    import warnings
    # Suppress numpy overflow warnings (we handle them in code)
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow.*')
    
    app = create_app()
    main_logger = logging.getLogger(__name__)
    # Use use_reloader=False on Windows to avoid thread issues, or use_reloader=True for development
    try:
        main_logger.info("Starting Flask server on http://127.0.0.1:5000")
        app.run(debug=True, port=5000, use_reloader=False)
    except KeyboardInterrupt:
        main_logger.info("Server stopped by user")
    except Exception as e:
        main_logger.error(f"Server error: {str(e)}")
        raise