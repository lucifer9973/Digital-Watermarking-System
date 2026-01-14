#!/usr/bin/env python3
"""
Production startup script for Digital Watermarking System
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.app import create_app

if __name__ == '__main__':
    app = create_app()

    # Get configuration from environment
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    debug = os.getenv('FLASK_ENV') == 'development'

    print(f"Starting Digital Watermarking System on {host}:{port}")
    print(f"Environment: {'Development' if debug else 'Production'}")

    app.run(host=host, port=port, debug=debug)
