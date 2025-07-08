from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from utils.deepface_utils import extract_embedding
from utils.tenseal_context import load_secret_context
from utils.face_utils import check_face_completeness
import numpy as np
import tenseal as ts
import requests
import os
import cv2
from deepface import DeepFace

router = APIRouter()
SERVER_URL = os.getenv("SERVER_URL")

@router.post("/register-face/")
async def register_face(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        # Step 1: Face detection
        img = cv2.imread(temp_path)
        if img is None:
            raise HTTPException(
                status_code=400,
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
                    status_code=400,
                    detail="No face detected in the image. Please ensure your face is clearly visible and try again."
                )
            raise

        if not face_objs or len(face_objs) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image. Please ensure your face is clearly visible and try again."
            )

        # Step 2: Completeness check
        is_complete, error_message = check_face_completeness(face_objs[0], img)
        if not is_complete:
            raise HTTPException(
                status_code=400,
                detail=f"Incomplete face detected: {error_message}. Please ensure your entire face is visible and centered in the frame."
            )


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
                status_code=400,
                detail="Potential spoofing detected. Please use a real face for registration."
            )
          

        # Step 4: Extract embedding and register
        embedding = extract_embedding(temp_path)
        embedding_np = np.array(embedding, dtype=np.float32)
        context = load_secret_context()
        enc_vec = ts.ckks_vector(context, embedding_np)
        enc_bytes = enc_vec.serialize()
        files = {'file': ('embedding.bin', enc_bytes)}
        data = {'user_id': user_id}

        resp = requests.post(f"{SERVER_URL}/fhe/store-encrypted-embedding/", data=data, files=files)
        resp.raise_for_status()
        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")
    finally:
        os.remove(temp_path)