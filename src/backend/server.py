import base64
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from PIL import Image, ImageOps

from lib.utils.path import model_path
from models.visoin_regressor import VisionRegressor
from run.predict import get_eval_transform


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = VisionRegressor(model_name='convnext_tiny')
    model.load_best_model(model_path() / 'best_model_conv.pth')
    model.set_transform(get_eval_transform())
    model.model.eval()

    app.state.model = model
    yield


app = FastAPI(
    title='VisionRegressor Inference API',
    lifespan=lifespan,
)


@app.get('/')
def root():
    return 'Welcome'


@app.get('/health')
def health():
    return {'status': 'ok'}


def encode_rgb_image_to_base64(img: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        '.png',
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise ValueError('PNG 인코딩 실패')

    return base64.b64encode(encoded.tobytes()).decode('utf-8')


@app.post('/predict')
async def predict(req: Request, file: Annotated[UploadFile, File(...)]):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='이미지 파일만 업로드 가능합니다.')

    try:
        contents = await file.read()
        image = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert('RGB')

        model = req.app.state.model
        pred_angle = model.predict_pil(image)

        image = image.resize((224, 224))

        rgb_img = np.array(image).astype(np.float32) / 255.0
        input_tensor = app.state.model.transform(image).unsqueeze(0)

        result = req.app.state.model.generate_grad_cam(
            x=input_tensor,
            rgb_img=rgb_img,
        )

        grad_cam_b64 = encode_rgb_image_to_base64(result['cam_image'])

        # png_bytes = encode_rgb_image_to_png_bytes(result['cam_image'])

        return {
            'filename': file.filename,
            'predicted_angle': float(pred_angle),
            'unit': 'degree',
            'grad_cam_img': grad_cam_b64,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f'추론 중 오류 발생: {str(e)}'
        ) from e


def encode_rgb_image_to_png_bytes(img: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(
        '.png',
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise ValueError('PNG 인코딩 실패')
    return encoded.tobytes()


@app.post('/grad-cam', response_class=Response)
async def grad_cam(req: Request, file: Annotated[UploadFile, File(...)]):
    try:
        contents = await file.read()
        image = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert('RGB')

        # 모델 입력 크기에 맞춤
        image = image.resize((224, 224))

        rgb_img = np.array(image).astype(np.float32) / 255.0
        input_tensor = app.state.model.transform(image).unsqueeze(0)

        result = req.app.state.model.generate_grad_cam(
            x=input_tensor,
            rgb_img=rgb_img,
        )

        png_bytes = encode_rgb_image_to_png_bytes(result['cam_image'])

        return Response(content=png_bytes, media_type='image/png')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
