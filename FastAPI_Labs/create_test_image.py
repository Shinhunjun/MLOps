import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml

# MNIST 데이터에서 샘플 이미지 생성
def create_test_images():
    # MNIST 데이터 로드
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data, mnist.target.astype(int)
    
    # 각 숫자(0-9)의 첫 번째 샘플 저장
    for digit in range(10):
        # 해당 숫자의 첫 번째 샘플 찾기
        digit_indices = np.where(y == digit)[0]
        if len(digit_indices) > 0:
            sample = X[digit_indices[0]]
            
            # 28x28로 reshape
            image_array = sample.reshape(28, 28)
            
            # PIL Image로 변환
            image = Image.fromarray(image_array.astype(np.uint8))
            
            # 파일로 저장
            filename = f"test_digit_{digit}.png"
            image.save(filename)
            print(f"생성됨: {filename}")

if __name__ == "__main__":
    create_test_images()
    print("테스트 이미지 생성 완료!")
