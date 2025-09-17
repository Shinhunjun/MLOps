import numpy as np
from scipy import ndimage
from skimage.transform import rotate, resize
from skimage.util import random_noise
import random

def augment_image(image, augmentation_factor=3):
    """
    MNIST 이미지에 데이터 증강을 적용합니다.
    
    Args:
        image (numpy.ndarray): 28x28 MNIST 이미지
        augmentation_factor (int): 증강할 이미지 개수
    
    Returns:
        list: 증강된 이미지들의 리스트
    """
    augmented_images = [image.copy()]  # 원본 이미지 포함
    
    for _ in range(augmentation_factor):
        aug_image = image.copy()
        
        # 1. 회전 (-15도 ~ +15도)
        if random.random() < 0.7:  # 70% 확률로 회전 적용
            angle = random.uniform(-15, 15)
            aug_image = rotate(aug_image, angle, mode='constant', cval=0, preserve_range=True)
        
        # 2. 이동 (최대 2픽셀)
        if random.random() < 0.6:  # 60% 확률로 이동 적용
            shift_x = random.uniform(-2, 2)
            shift_y = random.uniform(-2, 2)
            aug_image = ndimage.shift(aug_image, [shift_y, shift_x], mode='constant', cval=0)
        
        # 3. 확대/축소 (0.9 ~ 1.1배)
        if random.random() < 0.5:  # 50% 확률로 스케일 적용
            scale = random.uniform(0.9, 1.1)
            # 중앙에서 스케일링
            center = 14
            aug_image = resize(aug_image, (int(28 * scale), int(28 * scale)), mode='constant', cval=0)
            # 28x28로 다시 리사이즈
            if aug_image.shape != (28, 28):
                aug_image = resize(aug_image, (28, 28), mode='constant', cval=0)
        
        # 4. 기울기 (shear) - 최대 0.1 라디안
        if random.random() < 0.4:  # 40% 확률로 기울기 적용
            shear = random.uniform(-0.1, 0.1)
            # 간단한 기울기 변환
            rows, cols = aug_image.shape
            M = np.array([[1, shear, 0], [0, 1, 0]])
            aug_image = ndimage.affine_transform(aug_image, M, mode='constant', cval=0)
        
        # 5. 노이즈 추가 (가벼운 노이즈)
        if random.random() < 0.3:  # 30% 확률로 노이즈 적용
            noise_factor = random.uniform(0.01, 0.05)
            aug_image = random_noise(aug_image, var=noise_factor)
        
        # 6. 밝기 조정
        if random.random() < 0.4:  # 40% 확률로 밝기 조정
            brightness = random.uniform(0.8, 1.2)
            aug_image = np.clip(aug_image * brightness, 0, 1)
        
        # 값 범위 보정
        aug_image = np.clip(aug_image, 0, 1)
        
        augmented_images.append(aug_image)
    
    return augmented_images

def augment_dataset(X, y, augmentation_factor=3):
    """
    전체 데이터셋에 데이터 증강을 적용합니다.
    
    Args:
        X (numpy.ndarray): 입력 데이터 (N, 784)
        y (numpy.ndarray): 타겟 데이터 (N,)
        augmentation_factor (int): 각 이미지당 증강할 개수
    
    Returns:
        tuple: (증강된 X, 증강된 y)
    """
    print(f"원본 데이터: {X.shape[0]}개")
    
    augmented_X = []
    augmented_y = []
    
    for i in range(X.shape[0]):
        # 원본 이미지 추가
        augmented_X.append(X[i])
        augmented_y.append(y[i])
        
        # 28x28로 reshape
        image = X[i].reshape(28, 28)
        
        # 증강된 이미지들 생성
        aug_images = augment_image(image, augmentation_factor)
        
        # 원본 제외하고 증강된 이미지들만 추가
        for aug_image in aug_images[1:]:
            augmented_X.append(aug_image.flatten())
            augmented_y.append(y[i])
    
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    print(f"증강 후 데이터: {augmented_X.shape[0]}개 (x{augmentation_factor + 1}배)")
    
    return augmented_X, augmented_y

def visualize_augmentation(image, num_samples=8):
    """
    데이터 증강 결과를 시각화합니다.
    
    Args:
        image (numpy.ndarray): 28x28 이미지
        num_samples (int): 시각화할 증강 이미지 개수
    """
    import matplotlib.pyplot as plt
    
    augmented_images = augment_image(image, num_samples - 1)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, aug_img in enumerate(augmented_images):
        axes[i].imshow(aug_img, cmap='gray')
        axes[i].set_title(f'Augmented {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
