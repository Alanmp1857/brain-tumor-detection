from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8000"

MODEL = tf.keras.models.load_model("tumor_model_v2.h5")

CLASS_NAMES = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def resize_image(image, size=(150, 150)):
    """Resize the image to the given size"""
    img = Image.fromarray(image)
    img = img.resize(size)
    return np.array(img)


@app.post("/braintumor/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(await file.read())
        image = resize_image(image)
        img_batch = np.expand_dims(image, 0)

        json_data = {
            "instances": img_batch.tolist()
        }

        response = requests.post(endpoint, json=json_data)
        prediction = np.array(response.json()["predictions"][0])

        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        print(e)
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
