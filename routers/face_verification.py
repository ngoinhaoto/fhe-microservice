from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
import requests
import tenseal as ts
import numpy as np
from utils.deepface_utils import extract_embedding
from utils.tenseal_context import load_secret_context
import os
import tempfile
import base64
import logging

logging.basicConfig(
    level=logging.INFO,  # Or DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

router = APIRouter()
SERVER_URL = os.getenv("SERVER_URL")

# Set up logger
logger = logging.getLogger(__name__)

@router.post("/verify-face/")
async def verify_face(
    file: UploadFile = File(...),
    session_id: str = Form(None),
    request: Request = None
):
    temp_path = None
    try:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name

        # Extract embedding
        embedding = extract_embedding(temp_path)
        embedding_np = np.array(embedding, dtype=np.float32)
        context = load_secret_context()
        enc_vec = ts.ckks_vector(context, embedding_np)
        enc_bytes = enc_vec.serialize()

        # Prepare form data for server
        form_data = {
            "session_id": (None, session_id) if session_id else None,
            "file": ("embedding.bin", enc_bytes, "application/octet-stream"),
        }
        # Remove None values
        form_data = {k: v for k, v in form_data.items() if v is not None}

        # Send to server
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
        server_data["match_found"] = bool(best_match and highest_similarity > 0.5) #cosine threshold set over here

        return server_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FHE verification failed: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)