from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from typing import Optional
import tempfile
import os
import requests
import base64
import numpy as np
import tenseal as ts
from utils.deepface_utils import extract_embedding
from utils.tenseal_context import load_secret_context
from utils.face_utils import check_face_completeness
import os
import tempfile
import base64
import numpy as np
import logging
from deepface import DeepFace
import cv2


logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

COSINE_THRESHOLD = 0.5

router = APIRouter(tags=["Verification Operations"])
SERVER_URL = os.getenv("SERVER_URL")
logger = logging.getLogger(__name__)

@router.post("/verify-face/")
async def verify_face(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    request: Request = None
):
    temp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        logger.info(f"Image saved to temporary file: {temp_path}")


        img = cv2.imread(temp_path)
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to read image. Please upload a valid image file."
            )

        detector_to_use = "yunet" 
        try: 
        # Step 1: Extract faces to check if a face is detected
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=detector_to_use,
                    align=True
                )
            except Exception as e:
                # DeepFace may raise an exception if no face is detected
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

            # Step 2: Check completeness
            is_complete, error_message = check_face_completeness(face_objs[0], img)
            if not is_complete:
                raise HTTPException(
                    status_code=400,
                    detail=f"Incomplete face detected: {error_message}. Please ensure your entire face is visible and centered in the frame."
                )
            logger.info("Face completeness check passed")

            # Step 3: Anti-spoofing check (only if face was detected)
            try:
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
                        detail="Potential spoofing detected. Please use a real face for verification."
                    )
            except Exception as spoof_e:
                logger.warning(f"Anti-spoofing check failed or not available: {str(spoof_e)}")
                # If the error message is about no face, return the same as step 1
                if "face could not be detected" in str(spoof_e).lower() or "no face" in str(spoof_e).lower():
                    raise HTTPException(
                        status_code=400,
                        detail="No face detected in the image. Please ensure your face is clearly visible and try again."
                    )
                raise HTTPException(
                    status_code=400,
                    detail=f"Anti-spoofing failed: {str(spoof_e)}"
                )

        except HTTPException as http_ex:
            raise http_ex

        embedding = extract_embedding(temp_path)
        embedding_np = np.array(embedding, dtype=np.float32)
        context = load_secret_context()
        enc_vec = ts.ckks_vector(context, embedding_np)
        enc_bytes = enc_vec.serialize()

        form_data = {
            "session_id": (None, session_id) if session_id else None,
            "file": ("embedding.bin", enc_bytes, "application/octet-stream"),
        }
        form_data = {k: v for k, v in form_data.items() if v is not None}

        # Send to FHE server for verification
        server_resp = requests.post(
            f"{SERVER_URL}/fhe/verify-with-embedding/",
            files=form_data,
            timeout=60,
        )
        server_resp.raise_for_status()
        server_data = server_resp.json()

        context = load_secret_context()
        highest_similarity = float("-inf")
        best_match = None

        for result in server_data.get("results", []):
            try:
                enc_sim_bytes = base64.b64decode(result["encrypted_similarity"])
                enc_sim = ts.lazy_ckks_vector_from(enc_sim_bytes)
                enc_sim.link_context(context)
                sim_value = enc_sim.decrypt()[0]
                result["similarity"] = sim_value

                # Log the decrypted similarity for each user
                logger.info(
                    f"Decrypted similarity for user {result.get('user_id')}, "
                    f"{result.get('full_name')}: {sim_value:.4f}"
                )

                if sim_value > highest_similarity:
                    highest_similarity = sim_value
                    best_match = result
            except Exception as e:
                result["similarity"] = None
                result["decrypt_error"] = str(e)
                logger.error(
                    f"Error decrypting similarity for user {result.get('user_id')}: {e}"
                )

        server_data["highest_similarity"] = highest_similarity if highest_similarity != float("-inf") else None
        server_data["best_match"] = best_match
        server_data["match_found"] = bool(best_match and highest_similarity > COSINE_THRESHOLD)

        return server_data

    except HTTPException as http_ex:
        raise http_ex  # <-- This will return the correct status code and message
    except Exception as e:
        logger.error(f"Error in FHE direct verification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FHE direct verification failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Temporary file deleted")