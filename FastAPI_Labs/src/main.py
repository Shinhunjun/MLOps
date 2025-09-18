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

class FeedbackData(BaseModel):
    pixels: list[float]  # 784개 픽셀 값
    label: int  # 정확한 라벨 (0-9)

@app.post("/save-feedback")
async def save_feedback_endpoint(feedback_data: FeedbackData):
    """피드백 데이터를 저장하고 자동으로 트리거를 체크하는 엔드포인트"""
    try:
        import os
        import json
        import time
        from PIL import Image
        import subprocess
        
        # 데이터 저장
        save_dir = "../new_data"
        os.makedirs(save_dir, exist_ok=True)
        
        # 카운트 파일 로드/생성
        count_file = os.path.join(save_dir, "count.json")
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                count_data = json.load(f)
        else:
            count_data = {"current_count": 0, "sub_set_count": 0}
        
        # 현재 sub_set 폴더 경로
        current_sub_set = f"sub_set_{count_data['sub_set_count']}"
        current_sub_set_path = os.path.join(save_dir, current_sub_set)
        os.makedirs(current_sub_set_path, exist_ok=True)
        
        # 이미지 저장
        timestamp = int(time.time() * 1000)
        filename = f"{feedback_data.label}_{timestamp}.png"
        file_path = os.path.join(current_sub_set_path, filename)
        
        # 픽셀 데이터를 이미지로 변환하여 저장
        pixels = feedback_data.pixels
        image_array = np.array(pixels).reshape(28, 28)
        image_array = (image_array * 255).astype(np.uint8)
        processed_image = Image.fromarray(image_array, 'L')
        processed_image.save(file_path)
        
        # 메타데이터 업데이트 (sub_set 폴더에)
        metadata_path = os.path.join(current_sub_set_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        metadata.append({
            "filename": filename,
            "true_label": feedback_data.label,
            "created_at": timestamp,
            "source": "streamlit_feedback"
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 카운트 파일 로드/생성
        count_file = os.path.join(save_dir, "count.json")
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                count_data = json.load(f)
        else:
            count_data = {"current_count": 0, "sub_set_count": 0}
        
        # 현재 sub_set 폴더의 데이터 개수 확인
        current_sub_set = f"sub_set_{count_data['sub_set_count']}"
        current_sub_set_path = os.path.join(save_dir, current_sub_set)
        
        if not os.path.exists(current_sub_set_path):
            os.makedirs(current_sub_set_path, exist_ok=True)
        
        # 현재 sub_set의 파일 개수 확인
        current_files = [f for f in os.listdir(current_sub_set_path) if f.endswith('.png')]
        current_count = len(current_files)
        
        if current_count >= 10:
            # 10개가 모였으면 다음 sub_set으로 이동
            count_data['sub_set_count'] += 1
            count_data['current_count'] = 0
            
            # 카운트 파일 업데이트
            with open(count_file, 'w') as f:
                json.dump(count_data, f, indent=2)
            
            # 자동 트리거 실행
            result = subprocess.run(
                ["python", "trigger_retrain.py"], 
                capture_output=True, 
                text=True,
                cwd="../"
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"피드백이 저장되었습니다. ({current_count}개) 재훈련이 자동으로 트리거되었습니다! sub_set_{count_data['sub_set_count']-1}이 학습됩니다.",
                    "data_count": current_count,
                    "sub_set": f"sub_set_{count_data['sub_set_count']-1}",
                    "triggered": True
                }
            else:
                return {
                    "status": "success",
                    "message": f"피드백이 저장되었습니다. ({current_count}개) 하지만 트리거에 실패했습니다.",
                    "data_count": current_count,
                    "triggered": False,
                    "error": result.stderr
                }
        else:
            return {
                "status": "success",
                "message": f"피드백이 저장되었습니다. ({current_count}/10개) - sub_set_{count_data['sub_set_count']}",
                "data_count": current_count,
                "sub_set": f"sub_set_{count_data['sub_set_count']}",
                "triggered": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"피드백 저장 중 오류: {str(e)}")



    
