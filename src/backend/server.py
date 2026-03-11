from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps

from lib.utils.path import model_path
from models.visoin_regressor import VisionRegressor


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = VisionRegressor(model_name='convnext_tiny')
    model.load_checkpoint(model_path() / 'best_model_conv.pth')
    model.model.eval()

    app.state.model = model
    yield


app = FastAPI(
    title='VisionRegressor Inference API',
    lifespan=lifespan,
)

# 서버 시작 시 1회만 모델 로드


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
async def predict(req: Request, file: Annotated[UploadFile, File(...)]):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='이미지 파일만 업로드 가능합니다.')

    try:
        contents = await file.read()
        image = ImageOps.exif_transpose(Image.open(BytesIO(contents))).convert('RGB')

        model = req.app.state.model
        pred_angle = model.predict_pil(image)

        return {
            'filename': file.filename,
            'predicted_angle': float(pred_angle),
            'unit': 'degree',
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f'추론 중 오류 발생: {str(e)}'
        ) from e
