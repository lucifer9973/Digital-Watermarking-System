# Digital Watermarking System - DWT-SVD-GOA

A web-based digital watermarking system using Discrete Wavelet Transform (DWT), Singular Value Decomposition (SVD), and Grasshopper Optimization Algorithm (GOA).

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Windows/Linux/Mac OS

### Step 1: Activate Virtual Environment

If you already have a virtual environment set up (you do - I can see `venv` folder):

**On Windows:**
```Windows Terminal
.\venv\Scripts\Activate
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

If you get an execution policy error on Windows, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 2: Install Dependencies

Make sure all required packages are installed:

```bash
pip install -r requirements.txt
```

Or if that doesn't work, install individually:
```bash
pip install flask==2.3.3 numpy opencv-python==4.8.1.78 PyWavelets==1.4.1 scikit-image==0.22.0
```

### Step 3: Run the Application

Navigate to the backend directory and run the Flask app:

```bash
cd backend
python app.py
```

Or from the root directory:
```bash
python backend/app.py
```

The application will start on `http://127.0.0.1:5000` or `http://localhost:5000`

### Step 4: Access the Web Interface

Open your web browser and go to:
```
http://127.0.0.1:5000
```

You should see the Digital Watermarking web interface where you can:
- Upload a cover image (e.g., `lena.png`)
- Upload a watermark image (e.g., `logo.png`)
- Click "Embed Watermark" to process
- Download the watermarked image

## 📁 Project Structure

```
.
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── algorithms/            # Core algorithms
│   │   ├── dwt_transform.py
│   │   ├── svd_transform.py
│   │   ├── goa_optimizer.py
│   │   ├── watermark_embedding.py
│   │   └── watermark_extraction.py
│   ├── templates/
│   │   └── index.html        # Web interface
│   ├── utils/
│   │   └── metrics.py        # Quality metrics
│   └── routes/
│       └── watermark_routes.py
├── data/
│   ├── sample_images/        # Test images
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Troubleshooting

### Issue: Module not found errors
**Solution:** Make sure you're in the virtual environment and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution:** Change the port in `backend/app.py`:
```python
app.run(debug=True, port=5001)  # Use port 5001 instead
```

### Issue: Images not loading
**Solution:** Make sure image files are in supported formats (PNG, JPG, JPEG) and are not corrupted.

### Issue: "Embedding process failed" error
**Solution:** 
- Check that both images are valid
- Ensure images are not too large (recommended: 512x512 or smaller)
- Check the console/terminal for detailed error messages

## 🧪 Testing

Run tests to verify everything is working:

```bash
pytest tests/ -v
```

## 📝 Usage Example

1. **Prepare Images:**
   - Cover image: Use images from `data/sample_images/` (e.g., `lena.png`)
   - Watermark: Use images from `data/watermark_images/` (e.g., `logo.png`)

2. **Embed Watermark:**
   - Open the web interface
   - Select cover image
   - Select watermark image
   - Click "Embed Watermark"
   - Wait for processing (may take 10-30 seconds)
   - Download the result

3. **Extract Watermark:**
   - Use the extraction feature (if implemented in routes)
   - Provide watermarked image and original cover image
   - Extract the watermark

## 🔍 Key Features

- **DWT-SVD-GOA Algorithm:** Advanced watermarking using wavelet transforms and optimization
- **Modern Web Interface:** Sleek, responsive UI with dark/light theme toggle, smooth animations, and interactive elements
- **Enhanced User Experience:** Drag-and-drop file uploads, real-time previews, progress indicators, and mobile-responsive design
- **Quality Metrics:** PSNR, SSIM, MSE, NCC calculations with visual feedback
- **Optimization:** GOA-based optimal scaling factor selection with interactive strength controls

## 📚 Additional Resources

- See `Quick_Start_Guide.md` for detailed project documentation
- See `DWT_SVD_GOA_LLM_Prompts.md` for algorithm specifications
- See `Code_Templates_Examples.md` for code examples

## 🛠️ Development

### Running in Development Mode

The app runs in debug mode by default. For production, modify `backend/app.py`:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

### Adding New Features

1. Modify algorithms in `backend/algorithms/`
2. Update routes in `backend/routes/`
3. Update frontend in `backend/templates/index.html`

## 📄 License

This project is for educational purposes.

## 👥 Author

Digital Watermarking Final Year Project

---

**Need Help?** Check the error messages in the terminal/console for detailed debugging information.

