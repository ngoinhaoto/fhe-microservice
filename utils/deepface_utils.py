from deepface import DeepFace

def extract_embedding(image_path: str) -> list:
    embedding_objs = DeepFace.represent(
        img_path=image_path,
        model_name="VGG-Face",
        detector_backend="yunet",
        enforce_detection=True,
        align=True
    )
    if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
        return embedding_objs[0]["embedding"]
    raise ValueError("No face detected in the image.")