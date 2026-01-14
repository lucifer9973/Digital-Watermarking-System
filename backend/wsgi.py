import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.app import create_app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    app.run()
