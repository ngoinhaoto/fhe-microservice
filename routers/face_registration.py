from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from utils.deepface_utils import extract_embedding
from utils.tenseal_context import load_secret_context
import numpy as np
import tenseal as ts
import requests
import os

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")
    finally:
        os.remove(temp_path)