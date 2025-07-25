import tenseal as ts
import os
import base64
from dotenv import load_dotenv

load_dotenv()
CONTEXT_DIR = os.getenv("CONTEXT_DIR", "context")
os.makedirs(CONTEXT_DIR, exist_ok=True)

SECRET_PATH = os.path.join(CONTEXT_DIR, "secret.txt")
PUBLIC_PATH = os.path.join(CONTEXT_DIR, "public.txt")

def write_data(file_name, data):
    if isinstance(data, bytes):
        data = base64.b64encode(data)
    with open(file_name, "wb") as f:
        f.write(data)

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    return base64.b64decode(data)

def ensure_context():
    if not (os.path.exists(SECRET_PATH) and os.path.exists(PUBLIC_PATH)):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        ) #default conteext with p = 200
        context.generate_galois_keys()
        context.global_scale = 2**40

        secret_context = context.serialize(save_secret_key=True)
        write_data(SECRET_PATH, secret_context)

        context.make_context_public()
        public_context = context.serialize()
        write_data(PUBLIC_PATH, public_context)
        print(f"TenSEAL context generated and saved in {CONTEXT_DIR}/")
    else:
        print(f"TenSEAL context files found in {CONTEXT_DIR}/. Loading existing context.")

def load_secret_context():
    return ts.context_from(read_data(SECRET_PATH))

def load_public_context():
    return ts.context_from(read_data(PUBLIC_PATH))