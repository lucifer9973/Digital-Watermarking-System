# 🔐 Digital Watermarking System - DWT-SVD-GOA

A sophisticated web-based digital watermarking system using **Discrete Wavelet Transform (DWT)**, **Singular Value Decomposition (SVD)**, and **Grasshopper Optimization Algorithm (GOA)** for robust image watermarking and extraction.

![Digital Watermarking System](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green?style=flat-square)
![License](https://img.shields.io/badge/License-Educational-orange?style=flat-square)

## ✨ Features

- **🎨 Modern Web Interface** - Beautiful glassmorphic design with animated gradients
- **🖼️ DWT-SVD-GOA Algorithm** - Advanced watermarking using wavelet transforms and optimization
- **⚡ Real-time Processing** - Fast image processing with progress indicators
- **📤 Drag-and-Drop Upload** - Intuitive file upload with image previews
- **💾 Easy Download** - Download watermarked and extracted images instantly
- **📱 Responsive Design** - Works seamlessly on desktop and mobile devices
- **🔍 High-Quality Output** - PSNR, SSIM, MSE quality metrics
- **🚀 Optimized Performance** - GOA-based optimal scaling factor selection


## 🚀 Quick Start

### Prerequisites
- **Python** 3.8 or higher
- **pip** package manager
- **Windows/Linux/Mac OS** operating system

### Step 1: Clone or Navigate to Project

```bash
cd Digital-Watermarking-System
```

### Step 2: Create and Activate Virtual Environment

**On Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

**Option A - From Root Directory:**
```bash
.\.venv\Scripts\python.exe backend/app.py
```

**Option B - From Backend Directory:**
```bash
cd backend
python app.py
```

The application will start on:
```
http://127.0.0.1:5000
```

### Step 5: Access the Web Interface

Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📸 Screenshots & UI Overview

### Main Interface
The application features a modern, dark-themed interface with glassmorphic design:

**Header Section:**
- Animated gradient title with smooth color transitions
- Professional subtitle and tech stack display

**Embed Watermark Section:**
- Drag-and-drop cover image upload area
- Drag-and-drop watermark image upload area
- Image preview with selected filename display
- Process button with visual feedback
- Result display with alpha value and download option

**Extract Watermark Section:**
- Upload watermarked image
- Upload original cover image
- Alpha value input slider with real-time feedback
- Process button and result display
- Download extracted watermark functionality

### Color Scheme
- **Primary Colors:** Cyan (#00d4ff), Purple (#bb86fc), Pink (#ff6b9d)
- **Background:** Dark gradient with fixed positioning
- **Cards:** Glassmorphic effect with blur and transparency
- **Hover Effects:** Smooth scale and glow animations

---

## 💻 How to Use

### Embedding Watermark

1. **Select Cover Image**
   - Click the "Click or drag image here" area in the Embed section
   - Or drag and drop an image file
   - Supported formats: PNG, JPG, JPEG (up to 10MB)

2. **Select Watermark Image**
   - Select your watermark image (logo, text, pattern, etc.)
   - Image preview will display automatically
   - Watermark can be smaller than cover image

3. **Process**
   - Click the **"EMBED WATERMARK"** button
   - Processing typically takes 10-30 seconds
   - Progress indicator shows when processing is complete

4. **Download Result**
   - View the alpha value used in the watermarking process
   - Click **"DOWNLOAD"** to save the watermarked image
   - Image is downloaded as `watermarked_image.png`

### Extracting Watermark

1. **Upload Watermarked Image**
   - Provide the image containing the watermark
   - This should be the output from the embed process

2. **Upload Original Cover Image**
   - Provide the original image before watermarking
   - Used to extract the embedded watermark

3. **Set Alpha Value**
   - Use the slider or text input to set the alpha value
   - This should match the alpha used during embedding
   - Default value: 0.10 (adjustable from 0.01 to 1.0)

4. **Extract**
   - Click **"EXTRACT WATERMARK"** button
   - Extracted watermark image will be displayed
   - Download the extracted watermark

---

## 📁 Project Structure

```
Digital-Watermarking-System/
├── backend/
│   ├── app.py                           # Main Flask application
│   ├── wsgi.py                          # WSGI configuration for production
│   │
│   ├── algorithms/                      # Core watermarking algorithms
│   │   ├── dwt_transform.py            # Discrete Wavelet Transform
│   │   ├── svd_transform.py            # Singular Value Decomposition
│   │   ├── goa_optimizer.py            # Grasshopper Optimization Algorithm
│   │   ├── watermark_embedding.py      # Main embedding logic
│   │   ├── watermark_extraction.py     # Main extraction logic
│   │   └── __pycache__/
│   │
│   ├── routes/
│   │   └── watermark_routes.py         # API endpoints
│   │
│   ├── templates/
│   │   └── index.html                  # Modern web interface
│   │
│   ├── utils/
│   │   ├── metrics.py                  # Quality metrics (PSNR, SSIM, MSE, NCC)
│   │   └── __pycache__/
│   │
│   └── __pycache__/
│
├── data/
│   └── sample_images/                   # Sample test images
│
├── Dockerfile                           # Docker configuration
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
├── setup.py                             # Setup configuration
└── vercel.json                          # Vercel deployment config
```

---

## 🔧 Technical Details

### Algorithm Overview

**DWT-SVD-GOA Approach:**

1. **DWT (Discrete Wavelet Transform)**
   - Decomposes images into frequency components
   - Separates low and high-frequency information
   - LL band used for watermark embedding (more robust)

2. **SVD (Singular Value Decomposition)**
   - Further decomposes the LL band
   - Modifies singular values for watermark insertion
   - Provides robustness against image attacks

3. **GOA (Grasshopper Optimization Algorithm)**
   - Finds optimal scaling factor (alpha)
   - Balances imperceptibility vs. robustness
   - Machine learning-based optimization

### Dependencies

```
flask==2.3.3                    # Web framework
flask-cors==4.0.0               # CORS support
flask-talisman==1.1.0           # Security headers
python-dotenv==1.0.0            # Environment variables
gunicorn==21.2.0                # Production server
numpy>=1.26                     # Numerical computing
opencv-python>=4.8              # Image processing
PyWavelets>=1.5                 # Wavelet transforms
scikit-image>=0.22              # Advanced image processing
scipy>=1.17                     # Scientific computing
pytest==7.4.2                   # Testing framework
pytest-flask==1.3.0             # Flask testing
Werkzeug==2.3.7                 # WSGI utilities
```

---

## 🧪 Troubleshooting

| Issue | Solution |
|-------|----------|
| **ModuleNotFoundError** | Activate virtual environment and run `pip install -r requirements.txt` |
| **Port 5000 already in use** | Change port in `backend/app.py` (line: `app.run(port=5001)`) |
| **Images not loading** | Ensure images are PNG/JPG and not corrupted (max 10MB) |
| **"Embedding failed" error** | Check console for detailed error; ensure images aren't too large (512x512 recommended) |
| **Slow processing** | Larger images take longer; try resizing to 512x512 pixels |
| **CORS errors** | Already configured in Flask app; check browser console for details |
| **Permission denied on Windows** | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## 📊 Performance Metrics

The system calculates quality metrics for watermarked images:

- **PSNR (Peak Signal-to-Noise Ratio)** - Higher is better (>30dB is good)
- **SSIM (Structural Similarity Index)** - Measures visual quality (0-1, higher is better)
- **MSE (Mean Squared Error)** - Difference between images (lower is better)
- **NCC (Normalized Cross Correlation)** - Watermark similarity (0-1, higher is better)

---

## 🚀 Deployment

### Local Production Run

```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:create_app()
```

### Docker Deployment

```bash
docker build -t watermarking-system .
docker run -p 5000:5000 watermarking-system
```

### Vercel Deployment

```bash
vercel deploy
```

---

## 📝 Example Usage


## 📝 Example Usage

### Step-by-Step Watermarking Process

**1. Prepare Test Images**
- Use any image as cover image (landscape, portrait, etc.)
- Use a smaller image as watermark (logo, QR code, signature, etc.)

**2. Embed Watermark**
`ash
# Open http://127.0.0.1:5000 in browser
# 1. Drag and drop cover image
# 2. Drag and drop watermark image
# 3. Click "EMBED WATERMARK"
# 4. Download the watermarked image
`

**3. Extract Watermark**
`ash
# 1. Upload the previously watermarked image
# 2. Upload the original cover image
# 3. Enter the alpha value (shown after embedding)
# 4. Click "EXTRACT WATERMARK"
# 5. Download the extracted watermark
`

---

## 🔬 Algorithm Explanation

### Watermark Embedding Process

1. **Image Decomposition (DWT)**
   - Apply multi-level wavelet decomposition
   - Extract LL (low-frequency) band
   - High-frequency info preserved in other bands

2. **Value Decomposition (SVD)**
   - Apply SVD to LL band: LL = U × S × V^T
   - Modify singular values: S' = S + α × W
   - α = scaling factor, W = watermark matrix

3. **Optimization (GOA)**
   - Find optimal α value using grasshopper algorithm
   - Maximize imperceptibility (visual quality)
   - Maintain robustness against attacks

4. **Reconstruction**
   - Inverse SVD: LL' = U × S' × V^T
   - Inverse DWT to get watermarked image
   - Maintain quality metrics above threshold

### Watermark Extraction Process

1. **Decompose watermarked image** using same DWT-SVD process
2. **Extract watermark matrix**: W' = (S' - S) / α
3. **Reconstruct watermark image** from extracted matrix
4. **Apply quality metrics** to verify extraction

---

## 🎓 Educational Value

This project demonstrates:
- **Signal Processing:** DWT fundamentals
- **Linear Algebra:** SVD applications
- **Optimization:** Nature-inspired algorithms (GOA)
- **Web Development:** Flask full-stack application
- **Image Processing:** Advanced CV techniques
- **Software Engineering:** Clean code, modularity, testing

---

## 🔐 Security Features

- **Blind Extraction:** Can extract watermark without original image (with alpha)
- **Robust:** Resistant to common attacks (compression, noise, rotation)
- **Imperceptible:** Visual quality preserved after watermarking
- **Scalable:** Works with various image sizes and formats

---

## 📞 Support & FAQ

**Q: What image formats are supported?**
A: PNG, JPG, JPEG (up to 10MB)

**Q: What's the recommended image size?**
A: 512×512 pixels for optimal performance

**Q: How long does processing take?**
A: 10-30 seconds depending on image size

**Q: Can I use color images?**
A: Yes, but they're converted to grayscale for processing

**Q: What alpha value should I use?**
A: Default 0.1 works well; higher values = stronger watermark (less visible)

**Q: Can I recover watermark without original image?**
A: No, extraction requires original cover image for comparison

---

## 🛠️ Development & Contribution

### Setting Up Development Environment

`ash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-flask

# Run tests
pytest -v
`

### Code Structure

- **backend/algorithms/** - Core mathematical algorithms
- **backend/routes/** - API endpoint definitions
- **backend/templates/** - Frontend HTML/CSS/JavaScript
- **backend/utils/** - Utility functions and metrics

### Making Changes

1. Modify algorithm files in ackend/algorithms/
2. Update routes if needed in ackend/routes/
3. Test changes locally
4. Update 
equirements.txt if adding new packages

---

## 📄 License

This project is for **educational purposes** in a Final Year academic project.

---

## 👥 Authors & Credits

**Digital Watermarking System**
- Final Year Academic Project
- Built with Python, Flask, and modern web technologies
- Implements cutting-edge watermarking algorithms

---

## 📚 References & Resources

- DWT: [PyWavelets Documentation](https://pywavelets.readthedocs.io/)
- SVD: [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- Image Processing: [OpenCV Docs](https://docs.opencv.org/)
- Flask: [Flask Official Docs](https://flask.palletsprojects.com/)

---

**Last Updated:** January 2026
**Status:** ✅ Production Ready

---

### 🚀 Quick Command Reference

\\\ash
# Activate environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run application
python backend/app.py

# Access web interface
http://127.0.0.1:5000
\\\

---

**🎉 Thank you for using the Digital Watermarking System!**
