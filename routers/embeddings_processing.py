from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import requests
import os
from dotenv import load_dotenv
from utils.tenseal_context import load_secret_context
import tenseal as ts
import numpy as np
import base64
from utils.deepface_utils import extract_embedding

load_dotenv()
MAIN_SERVER_URL = os.getenv("SERVER_URL") + "/api/face/register-embedding/"

router = APIRouter()

@router.post("/encrypt")
async def encrypt_embedding(file: UploadFile = File(...)):
    embedding_bytes = await file.read()
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    context = load_secret_context()
    enc_vec = ts.ckks_vector(context, embedding)
    enc_bytes = enc_vec.serialize()
    # Base64 encode for safe transport
    return {"encrypted": base64.b64encode(enc_bytes).decode("utf-8")}


@router.post("/extract-embedding")
async def extract_embedding_route(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    try:
        embedding = extract_embedding(temp_path)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(temp_path)



@router.get("/test-fhe-roundtrip")
def test_fhe_roundtrip():
    # 1. Generate random embedding
    embedding = np.random.rand(512).astype(np.float32)
    embedding_bytes = embedding.tobytes()

    # 2. Encrypt with secret context
    context = load_secret_context()
    enc_vec = ts.ckks_vector(context, embedding)
    enc_bytes = enc_vec.serialize()

    # 3. Send to server for similarity test (server must have a test endpoint)
    files = {'file': ('embedding.bin', enc_bytes)}
    resp = requests.post(f"{MAIN_SERVER_URL}/test-similarity/", files=files)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Server FHE test failed")
    enc_result_bytes = resp.content

    # 4. Decrypt result
    enc_result = ts.lazy_ckks_vector_from(enc_result_bytes)
    enc_result.link_context(context)
    decrypted = enc_result.decrypt()[0]

    # 5. Compute expected (dot product with itself)
    expected = float(np.dot(embedding, embedding))
    print(f"Decrypted: {decrypted:.6f}, Expected: {expected:.6f}")
    return {"decrypted": decrypted, "expected": expected, "error": abs(decrypted - expected)}