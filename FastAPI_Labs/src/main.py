from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
import numpy as np
from predict import predict_with_cnn, reload_models


app = FastAPI()

class MNISTData(BaseModel):
    pixels: list[float]  # 784개 픽셀 값 (28x28 = 784)
    
    class Config:
        # 784개 픽셀 검증
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784  # 784개 0.0 값 예시 (정규화된 검은 이미지)
            }
        }

class MNISTResponse(BaseModel):
    prediction: int  # 0-9 숫자 예측 결과
    confidence: float  # 예측 신뢰도 (선택사항)

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=MNISTResponse)
async def predict_mnist(mnist_data: MNISTData):
    """CNN 모델을 사용한 예측"""
    try:
        # 784개 픽셀을 2D 배열로 변환 (1, 784)
        features = np.array([mnist_data.pixels])
        
        # 픽셀 개수 검증
        if len(mnist_data.pixels) != 784:
            raise HTTPException(
                status_code=400, 
                detail="MNIST 이미지는 정확히 784개 픽셀(28x28)이어야 합니다"
            )
        
        # 픽셀 값 범위 검증 (0-1, 정규화된 값)
        if not all(0 <= pixel <= 1 for pixel in mnist_data.pixels):
            raise HTTPException(
                status_code=400,
                detail="픽셀 값은 0-1 범위(정규화된 값)여야 합니다"
            )

        # CNN 모델로 예측
        predictions, confidences = predict_with_cnn(features)
        predicted_class = int(predictions[0])
        confidence_score = float(confidences[0])
        
        return MNISTResponse(
            prediction=predicted_class,
            confidence=confidence_score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-models")
async def reload_models_endpoint():
    """모델을 다시 로드하는 엔드포인트 (GitHub Actions에서 호출)"""
    try:
        reload_models()
        return {"status": "success", "message": "모델이 성공적으로 다시 로드되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패: {str(e)}")


    
