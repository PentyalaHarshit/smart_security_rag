AI SMART SECURITY CAMERA WITH RAG + LOCAL LLM

PROJECT OVERVIEW

This project is an AI-based smart security system that combines:

Computer Vision (OpenCV + YOLO)
Event Logging
Retrieval-Augmented Generation (RAG)
Local Large Language Model (Ollama)

The system detects objects from a camera, stores events, and allows users to ask questions in natural language.

FEATURES

Real-time object detection using YOLO
Automatic logging of detected events
Multi-document support (TXT and PDF files)
Semantic search using ChromaDB
Local LLM (no API required)
FastAPI web interface
Fully offline capable

PROJECT STRUCTURE

smart_security_rag/

app.py
camera_detect.py

data/
    events.txt
    docs/
        security_rules.txt
        object_notes.txt

INSTALLATION

Step 1: Install Python dependencies

pip install opencv-python ultralytics chromadb sentence-transformers fastapi uvicorn python-multipart pypdf ollama

Step 2: Install Ollama

Step 3: Download LLM model

ollama pull llama3

OR (faster option)

ollama pull mistral

HOW TO RUN

Step 1: Start camera detection

python camera_detect.py

Camera opens
Objects are detected
Events are saved in events.txt
Press 'q' to stop

Step 2: Start FastAPI server

uvicorn app:app --reload

Step 3: Open in browser

http://127.0.0.1:8000

EXAMPLE QUESTIONS

Was a person detected?
Was phone detected?
Summarize camera events
Is there any security risk?
What valuable objects were seen?