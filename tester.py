from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import io
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import sqlite3
from datetime import datetime, timedelta
import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    DATABASE_PATH = 'employee_faces.db'
    FACE_MODEL = "resnet50_2020-07-20"
    MAX_IMAGE_SIZE = 1048
    DEVICE = "cpu"  # Change to "cuda" if GPU available
    SIMILARITY_THRESHOLD = 0.5
    MAX_FACES_PER_IMAGE = 10

config = Config()

class FaceRecognitionSystem:
    def __init__(self):
        self.model = None
        self.load_model()
        self.init_database()
    
    def load_model(self):
        """Load the RetinaFace model"""
        try:
            self.model = get_model(
                config.FACE_MODEL, 
                max_size=config.MAX_IMAGE_SIZE, 
                device=config.DEVICE
            )
            self.model.eval()
            logger.info("RetinaFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def init_database(self):
        """Initialize SQLite database for employee faces"""
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Employees table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                department TEXT,
                position TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Face embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        # Attendance logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT NOT NULL,
                check_in_time TIMESTAMP,
                check_out_time TIMESTAMP,
                date DATE,
                status TEXT DEFAULT 'present',
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    
    def extract_face_features(self, image: np.ndarray) -> List[Dict]:
        """Extract face features using RetinaFace"""
        try:
            with torch.no_grad():
                annotations = self.model.predict_jsons(image)
            
            if not annotations or not annotations[0].get("bbox"):
                return []
            
            faces = []
            for annotation in annotations[:config.MAX_FACES_PER_IMAGE]:
                bbox = annotation["bbox"]
                landmarks = annotation.get("landmarks", [])
                
                # Extract face region
                x1, y1, x2, y2 = map(int, bbox)
                face_region = image[y1:y2, x1:x2]
                
                # Generate face embedding (simplified - in production use FaceNet or similar)
                face_embedding = self._generate_face_embedding(face_region)
                
                faces.append({
                    "bbox": bbox,
                    "landmarks": landmarks,
                    "embedding": face_embedding,
                    "face_region": face_region
                })
            
            return faces
        except Exception as e:
            logger.error(f"Failed to extract face features: {str(e)}")
            return []
    
    def _generate_face_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """Generate face embedding (simplified version)"""
        # In production, use FaceNet, ArcFace, or similar for embeddings
        # This is a simplified approach using image statistics
        try:
            if face_region.size == 0:
                return np.zeros(512)
            
            # Resize face to standard size
            face_resized = cv2.resize(face_region, (112, 112))
            
            # Convert to grayscale and normalize
            if len(face_resized.shape) == 3:
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
            else:
                face_gray = face_resized
            
            face_normalized = face_gray.astype(np.float32) / 255.0
            
            # Create a simple embedding using image patches
            embedding = []
            patch_size = 8
            for i in range(0, 112, patch_size):
                for j in range(0, 112, patch_size):
                    patch = face_normalized[i:i+patch_size, j:j+patch_size]
                    embedding.extend([
                        np.mean(patch),
                        np.std(patch),
                        np.min(patch),
                        np.max(patch)
                    ])
            
            return np.array(embedding[:512])  # Ensure fixed size
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return np.zeros(512)
    
    def register_employee(self, employee_data: dict, image) -> dict:
        """
        Register a new employee with their face.

        Parameters:
            employee_data (dict): Employee details (employee_id, name, etc.)
            image (PIL.Image or np.ndarray): Raw image of the employee

        Returns:
            dict: Result with success status and message
        """
        try:
            # If image is bytes, convert to PIL.Image
            if isinstance(image, bytes):
                from PIL import Image
                from io import BytesIO
                image = Image.open(BytesIO(image))

            # Extract face features directly from the raw image
            faces = self.extract_face_features(image)

            if not faces:
                return {"success": False, "message": "No face detected in image"}

            if len(faces) > 1:
                return {"success": False, "message": "Multiple faces detected. Please use image with single face"}

            face = faces[0]
            embedding = face["embedding"]

            # Store in database
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()

            # Insert employee details
            cursor.execute('''
                INSERT OR REPLACE INTO employees (employee_id, name, department, position)
                VALUES (?, ?, ?, ?)
            ''', (
                employee_data["employee_id"],
                employee_data["name"],
                employee_data.get("department", ""),
                employee_data.get("position", "")
            ))

            # Insert face embedding as blob
            embedding = face["embedding"].astype(np.float32) 
            embedding_blob = embedding.tobytes()
            cursor.execute('''
                INSERT INTO face_embeddings (employee_id, embedding)
                VALUES (?, ?)
            ''', (employee_data["employee_id"], embedding_blob))

            conn.commit()
            conn.close()

            return {
                "success": True,
                "message": f"Employee {employee_data['name']} registered successfully",
                "employee_id": employee_data["employee_id"]
            }

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to register employee: {str(e)}")
            return {"success": False, "message": str(e)}

 
    def recognize_employee(self, image) -> dict:
        """
        Recognize employee(s) from a raw image.

        Parameters:
            image (np.ndarray or PIL.Image or bytes): Input image

        Returns:
            dict: Recognition result
        """
        try:

            if isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")

            if isinstance(image, Image.Image):
                image = np.array(image)

            # Extract face features
            faces = self.extract_face_features(image)

            if not faces:
                return {"success": False, "message": "No face detected"}

            # Get all registered embeddings
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT fe.employee_id, fe.embedding, e.name
                FROM face_embeddings fe
                JOIN employees e ON fe.employee_id = e.employee_id
            ''')
            registered_faces = cursor.fetchall()
            conn.close()

            if not registered_faces:
                return {"success": False, "message": "No registered employees found"}

            best_matches = []
            for face in faces:
                query_embedding = face["embedding"]
                best_similarity = 0
                best_match = None

                for emp_id, embedding_blob, name in registered_faces:
                    # stored_embedding = np.frombuffer(embedding_blob, dtype=np.float64)
                    stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)


                    # Calculate cosine similarity
                    similarity = self._calculate_similarity(query_embedding, stored_embedding)
                    print(similarity, "this is similiarity")
                    print("stored embedding", stored_embedding.dtype)
                    print("query embedding", query_embedding.dtype)

                    if similarity > best_similarity and similarity > config.SIMILARITY_THRESHOLD:
                        best_similarity = similarity
                        best_match = {
                            "employee_id": emp_id,
                            "name": name,
                            "confidence": similarity,
                            "bbox": face["bbox"]
                        }

                if best_match:
                    best_matches.append(best_match)

            return {
                "success": True,
                "matches": best_matches,
                "total_faces": len(faces),
                "recognized_faces": len(best_matches)
            }

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to recognize employee: {str(e)}")
            return {"success": False, "message": str(e)}

    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure same length
            min_len = min(len(embedding1), len(embedding2))
            emb1 = embedding1[:min_len]
            emb2 = embedding2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def log_attendance(self, employee_id: str, action: str, confidence: float = 0.0) -> Dict:
        """Log attendance (check-in/check-out)"""
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            current_time = datetime.now()
            current_date = current_time.date()
            
            if action == "check_in":
                # Check if already checked in today
                cursor.execute('''
                    SELECT id FROM attendance_logs 
                    WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL
                ''', (employee_id, current_date))
                
                if cursor.fetchone():
                    conn.close()
                    return {"success": False, "message": "Already checked in today"}
                
                # Insert new attendance record
                cursor.execute('''
                    INSERT INTO attendance_logs (employee_id, check_in_time, date, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (employee_id, current_time, current_date, confidence))
                
            elif action == "check_out":
                # Find today's attendance record
                cursor.execute('''
                    SELECT id FROM attendance_logs 
                    WHERE employee_id = ? AND date = ? AND check_in_time IS NOT NULL AND check_out_time IS NULL
                ''', (employee_id, current_date))
                
                record = cursor.fetchone()
                if not record:
                    conn.close()
                    return {"success": False, "message": "No check-in record found for today"}
                
                # Update with check-out time
                cursor.execute('''
                    UPDATE attendance_logs 
                    SET check_out_time = ?, confidence = ?
                    WHERE id = ?
                ''', (current_time, confidence, record[0]))
            
            conn.commit()
            conn.close()
            
            return {"success": True, "message": f"Successfully {action.replace('_', '-')}ed"}
            
        except Exception as e:
            logger.error(f"Failed to log attendance: {str(e)}")
            return {"success": False, "message": str(e)}

# Initialize the face recognition system
face_system = FaceRecognitionSystem()

try:
    pil_image = Image.open("employee.jpg").convert("RGB")
    image_array = np.array(pil_image)
except Exception as e:
    print(f"Failed to load image: {str(e)}")

face_system.recognize_employee(image_array)
