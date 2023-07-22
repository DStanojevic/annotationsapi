import uvicorn
from fastapi import FastAPI, HTTPException
import fastapi.responses as responses
from fastapi.middleware.cors import CORSMiddleware
import torch
from segment_anything import sam_model_registry, SamPredictor

from utils import time_it
from dtos import Square
import imagerepo
from prediction_service import predict_annotation

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='../assets/sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)

predictor = SamPredictor(sam)

current_image_id = None

app = FastAPI()

origins = [
    "http://localhost:4200",  # Angular, React, Vue.js development server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, you can customize this and list ["GET", "POST"]
    allow_headers=["*"],  # Allows all headers, you can customize this
)


def _set_current_image(image_id: str):
    global current_image_id
    current_image_id = image_id


def _get_current_image():
    global current_image_id
    current_image_id


@app.get("/")
async def root():
    return {"message": "Hello World"}


@time_it
@app.get("/images/{image_id}")
def get_image(image_id: str):
    image_data = imagerepo.get_image(image_id)
    if image_data:
        predictor.set_image(image_data['content'])
        _set_current_image(image_id)
        return responses.FileResponse(image_data['location'])

    raise HTTPException(status_code=404, detail=f'Image with ID: {image_id} does not exists.')


@app.post("/images/{image_id}/annotate")
def annotate(image_id: str, input_box: Square):
    if image_id != current_image_id:
        raise HTTPException(
            status_code=422,
            detail="Image id {image_id} is different from the current image {_get_current_image()} that is loaded."
        )
    predicted_annotation = predict_annotation(predictor=predictor, box=input_box)
    return predicted_annotation


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # input = {
    #     'topLeft': {
    #         'x': 1175,
    #         'y': 296
    #     },
    #     'bottomRight': {
    #         'x': 1242,
    #         'y': 379
    #     }
    # }
    # in_box = Square(**input)
    # _ = get_image('img1')
    # res = annotate('img1', in_box)