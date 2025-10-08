import os
import shutil
import uuid
import cv2
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from predictor import predict_pose, process_video

app = FastAPI(
    title="Head Pose Estimation API",
    description="Upload images or videos and get processed output with head pose axes.",
    version="1.0.0"
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["jpg", "jpeg", "png", "mp4", "avi", "mov"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use jpg, jpeg, png, mp4, avi, or mov.")

    unique_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}.{file_ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{unique_id}_output.{file_ext}")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if file_ext in ["jpg", "jpeg", "png"]:
            result = predict_pose(input_path)
            if result is None:
                raise HTTPException(status_code=400, detail="No face detected")
            cv2.imwrite(output_path, result)
            return FileResponse(output_path, media_type="image/jpeg", filename="pose_output.jpg")
        elif file_ext in ["mp4", "avi", "mov"]:
            processed = process_video(input_path, output_path)
            if processed is None:
                raise HTTPException(status_code=500, detail="Video processing failed")
            return FileResponse(output_path, media_type="video/mp4", filename="pose_output.mp4")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Delay file deletion to ensure VideoCapture is closed
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except PermissionError:
                print(f"Warning: Could not delete {input_path} due to PermissionError. It may still be in use.")
                pass

@app.get("/")
async def root():
    return {"message": "Head Pose Estimation API is running. Go to /docs for Swagger UI."}