#!/usr/bin/env python
"""Simple script to run the FastAPI server."""
import sys
import os
from pathlib import Path

# Set up the path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
