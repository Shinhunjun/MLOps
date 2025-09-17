#!/usr/bin/env python3
"""
테스트용 데이터를 생성하는 스크립트
"""
import os
import json
import numpy as np
from PIL import Image
import time

def create_test_data():
    """테스트용 데이터 10개 생성"""
    data_dir = "new_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 테스트 데이터 생성
    test_data = []
    
    for i in range(10):
        # 랜덤한 28x28 이미지 생성 (0-9 숫자)
        digit = i % 10
        image_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        
        # 간단한 패턴 추가 (숫자 모양)
        if digit == 0:
            image_array[5:23, 8:20] = 255  # 사각형
            image_array[7:21, 10:18] = 0   # 내부 구멍
        elif digit == 1:
            image_array[5:23, 12:16] = 255  # 세로선
        elif digit == 2:
            image_array[5:23, 8:20] = 255  # 사각형
            image_array[7:11, 10:18] = 0   # 상단 구멍
            image_array[15:19, 10:18] = 0  # 하단 구멍
        # ... 다른 숫자들도 비슷하게
        
        # 이미지 저장
        timestamp = int(time.time() * 1000) + i
        filename = f"{digit}_{timestamp}.png"
        filepath = os.path.join(data_dir, filename)
        
        image = Image.fromarray(image_array, 'L')
        image.save(filepath)
        
        # 메타데이터 추가
        test_data.append({
            "filename": filename,
            "true_label": digit,
            "created_at": timestamp,
            "source": "test_data"
        })
        
        print(f"✅ 테스트 데이터 생성: {filename}")
    
    # 메타데이터 저장
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"📁 {len(test_data)}개의 테스트 데이터가 생성되었습니다.")
    print(f"📄 메타데이터: {metadata_path}")

if __name__ == "__main__":
    create_test_data()
