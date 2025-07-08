from fastapi import FastAPI
from routers.embeddings_processing import router
from routers.face_registration import router as face_registration_router
from routers.face_verification import router as face_verification_router
from utils.tenseal_context import ensure_context
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

ensure_context()

app = FastAPI(
    title="FHE Microservice",
    description="Handles encrypted face embedding operations using TenSEAL.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend's URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(face_registration_router)
app.include_router(face_verification_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True, log_level="info")