import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from utils.tenseal_utils import load_public_context
import tenseal as ts
import base64

EMBEDDINGS_DIR = "storage/embeddings"
router = APIRouter()

def save_embedding(user_id, embedding_bytes):
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    with open(f"{EMBEDDINGS_DIR}/{user_id}.bin", "wb") as f:
        f.write(embedding_bytes)

def load_embedding(user_id):
    with open(f"{EMBEDDINGS_DIR}/{user_id}.bin", "rb") as f:
        return f.read()

def list_user_ids():
    return [f.split(".")[0] for f in os.listdir(EMBEDDINGS_DIR) if f.endswith(".bin")]

@router.post("/register-embedding/")
async def register_embedding(user_id: str = Form(...), file: UploadFile = File(...)):
    encrypted_data = await file.read()
    save_embedding(user_id, encrypted_data)
    return {"status": "registered", "user_id": user_id}

@router.post("/compare-embedding/")
async def compare_embedding(user_id: str = Form(...), file: UploadFile = File(...)):
    # Load stored embedding for user
    stored_bytes = load_embedding(user_id)
    uploaded_bytes = await file.read()
    context = load_public_context()
    # Deserialize vectors
    enc_stored = ts.lazy_ckks_vector_from(stored_bytes)
    enc_stored.link_context(context)
    enc_uploaded = ts.lazy_ckks_vector_from(uploaded_bytes)
    enc_uploaded.link_context(context)
    # Compute squared Euclidean distance (encrypted)
    diff = enc_stored - enc_uploaded
    enc_dist2 = diff.dot(diff)
    # Return encrypted result (client will decrypt)
    return {"enc_distance": base64.b64encode(enc_dist2.serialize()).decode()}