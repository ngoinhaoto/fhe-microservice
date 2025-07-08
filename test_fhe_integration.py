import requests
import numpy as np
import tenseal as ts
import os
import base64

MICROSERVICE_URL = "http://localhost:8002/encrypt"
SERVER_TEST_URL = "http://localhost:8000/attendance/test-similarity/"
CONTEXT_DIR = "context"
SECRET_PATH = os.path.join(CONTEXT_DIR, "secret.txt")

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)

def test_encrypt_compute_decrypt():
    # 1. Generate random embedding
    embedding = np.random.rand(512).astype(np.float32)
    embedding_bytes = embedding.tobytes()

    # 2. Encrypt using microservice
    files = {'file': ('embedding.bin', embedding_bytes)}
    resp = requests.post(MICROSERVICE_URL, files=files)
    assert resp.status_code == 200
    enc_bytes = base64.b64decode(resp.json()["encrypted"])

    # 3. Send encrypted embedding to server for computation
    files = {'file': ('embedding.bin', enc_bytes)}
    resp2 = requests.post(SERVER_TEST_URL, files=files)
    assert resp2.status_code == 200
    enc_result_bytes = base64.b64decode(resp2.json()["result"])

    # 4. Decrypt result using secret context
    secret_context = ts.context_from(read_data(SECRET_PATH))
    enc_result = ts.lazy_ckks_vector_from(enc_result_bytes)
    enc_result.link_context(secret_context)
    decrypted = enc_result.decrypt()[0]

    # 5. Compare with expected
    expected = float(np.dot(embedding, embedding))
    print(f"Decrypted: {decrypted:.6f}, Expected: {expected:.6f}")
    assert abs(decrypted - expected) < 1e-3

if __name__ == "__main__":
    test_encrypt_compute_decrypt()
    print("FHE integration test passed!")