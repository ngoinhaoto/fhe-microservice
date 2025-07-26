from fastapi import APIRouter, UploadFile, File, HTTPException, Form, status # Import status
from utils.deepface_utils import extract_embedding
from utils.tenseal_context import load_secret_context
from utils.face_utils import check_face_completeness
import numpy as np
import tenseal as ts
import requests
import os
import cv2
from deepface import DeepFace
import tempfile # Import tempfile for secure temporary file creation

router = APIRouter()
SERVER_URL = os.getenv("SERVER_URL")

@router.post("/register-face/")
async def register_face(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    temp_path: str = None # Initialize temp_path to None
    try:
        # Use tempfile.NamedTemporaryFile for secure temporary file creation
        # delete=False means we manually delete it in the finally block
        # suffix ensures it has a .jpg extension for DeepFace
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name # Get the actual path of the temporary file

        # Step 1: Face detection
        img = cv2.imread(temp_path)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, # Use status.HTTP_400_BAD_REQUEST for consistency
                detail="Failed to read image. Please upload a valid image file."
            )
        detector_to_use = "yunet"  # or your preferred detector

        try:
            face_objs = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=detector_to_use,
                align=True
            )
        except Exception as e:
            if "face could not be detected" in str(e).lower() or "no face" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in the image. Please ensure your face is clearly visible and try again."
                )
            raise

        if not face_objs or len(face_objs) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image. Please ensure your face is clearly visible and try again."
            )

        # Step 2: Completeness check
        is_complete, error_message = check_face_completeness(face_objs[0], img)
        if not is_complete:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Incomplete face detected: {error_message}. Please ensure your entire face is visible and centered in the frame."
            )

        # Step 3: Anti-spoofing check
        anti_spoof_faces = DeepFace.extract_faces(
            img_path=temp_path,
            anti_spoofing=True
        )
        is_spoof = True
        if anti_spoof_faces and len(anti_spoof_faces) > 0:
            face_obj = anti_spoof_faces[0]
            is_real = face_obj.get("is_real", False)
            if is_real:
                is_spoof = False
        if is_spoof:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Potential spoofing detected. Please use a real face for registration."
            )
          
        # Step 4: Extract embedding and encrypt with TenSEAL
        embedding = extract_embedding(temp_path)
        embedding_np = np.array(embedding, dtype=np.float32)
        
        # Load secret context (consider loading this once globally if static)
        context = load_secret_context() 
        
        enc_vec = ts.ckks_vector(context, embedding_np)
        enc_bytes = enc_vec.serialize() # TenSEAL's built-in serialization

        # Step 5: Send encrypted embedding to the server
        files = {'file': ('embedding.bin', enc_bytes)}
        data = {'user_id': user_id}

        try:
            resp = requests.post(f"{SERVER_URL}/fhe/store-encrypted-embedding/", data=data, files=files)
            resp.raise_for_status() # This will raise HTTPError for 4xx/5xx responses
            return resp.json()
        except requests.exceptions.HTTPError as http_err: # <--- CATCH SPECIFICALLY HTTPError
            # Attempt to parse JSON detail from the main backend's response
            try:
                error_detail = http_err.response.json().get("detail", str(http_err))
            except ValueError: # If response is not JSON
                error_detail = str(http_err) # Fallback to generic HTTPError message
            
            # Re-raise as HTTPException with the specific status code and detail
            raise HTTPException(status_code=http_err.response.status_code, detail=error_detail)
        
    except HTTPException:
        raise # Re-raise FastAPI's HTTPException directly (e.g., from face detection, completeness, anti-spoofing checks)
    except Exception as e:
        # Catch any other truly unexpected errors and return a 500
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during registration: {str(e)}")
    finally:
        # Ensure the temporary file is deleted
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)