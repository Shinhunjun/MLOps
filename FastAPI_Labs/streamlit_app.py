import streamlit as st
import requests
import numpy as np
from PIL import Image
import io

# 페이지 설정
st.set_page_config(
    page_title="MNIST 숫자 인식기",
    page_icon="🔢",
    layout="wide"
)

# 제목
st.title("🔢 MNIST 숫자 인식기")
st.markdown("이미지를 업로드하면 AI가 숫자를 예측해드립니다!")

# FastAPI 엔드포인트 URL 생성 함수
def get_api_url(model_choice):
    """선택된 모델에 따라 적절한 API URL을 반환합니다."""
    base_url = api_url.replace("/predict", "")
    if model_choice == "XGBoost (FastAPI)":
        return f"{base_url}/predict"
    elif model_choice == "CNN (FastAPI)":
        return f"{base_url}/predict-cnn"
    else:
        return api_url

# 사이드바
st.sidebar.header("설정")
model_choice = st.sidebar.selectbox(
    "모델 선택",
    ["XGBoost (FastAPI)", "CNN (FastAPI)"],
    help="사용할 모델을 선택하세요"
)

api_url = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8000/predict",
    help="FastAPI 서버의 predict 엔드포인트 URL"
)

# 메인 컨텐츠
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 이미지 업로드")
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "숫자 이미지를 업로드하세요",
        type=['png', 'jpg', 'jpeg'],
        help="28x28 픽셀 크기의 숫자 이미지를 업로드하세요"
    )
    
    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        # 이미지 전처리
        st.subheader("🔄 이미지 전처리")
        
        # 그레이스케일 변환
        if image.mode != 'L':
            image = image.convert('L')
            st.info("이미지를 그레이스케일로 변환했습니다.")
        
        # 28x28 크기로 리사이즈
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        st.info("이미지를 28x28 크기로 리사이즈했습니다.")
        
        # 픽셀 값 추출 및 보수적 전처리
        try:
            pixels = np.array(image).flatten()
            
            # 1. MNIST 표준 정규화: 0-1 범위 (이진화 전에 정규화)
            pixels_2d = pixels.reshape(28, 28)
            pixels_2d = pixels_2d / 255.0 if pixels_2d.max() > 1.0 else pixels_2d
            
            # # 2. 가우시안 블러 적용 (노이즈 제거)
            # from skimage.filters import gaussian, threshold_otsu
            # pixels_2d = gaussian(pixels_2d, sigma=0.5)
            
            # # 3. 적응적 이진화 (Otsu 방법) - 자연스러운 임계값
            # threshold = threshold_otsu(pixels_2d)
            # pixels_2d = pixels_2d > threshold
            # pixels_2d = pixels_2d.astype(float)
            
            # # 4. 모멘트 기반 중앙 정렬 및 회전 보정
            # from scipy import ndimage
            # from skimage.measure import moments
            
            # # 이진 이미지에서 모멘트 계산
            # m = moments(pixels_2d)
            # if m[0, 0] > 0:  # 0이 아닌 픽셀이 있는 경우
            #     # 무게 중심 계산
            #     cx = m[1, 0] / m[0, 0]
            #     cy = m[0, 1] / m[0, 0]
                
            #     # 중앙으로 이동 (14, 14가 중앙)
            #     shift_x = 14 - cx
            #     shift_y = 14 - cy
                
            #     # 평행 이동 적용
            #     pixels_2d = ndimage.shift(pixels_2d, [shift_y, shift_x], mode='constant', cval=0)
                
            #     # 회전 보정 (2차 모멘트 기반)
            #     mu20 = m[2, 0] / m[0, 0] - cx**2
            #     mu02 = m[0, 2] / m[0, 0] - cy**2
            #     mu11 = m[1, 1] / m[0, 0] - cx * cy
                
            #     if mu20 != mu02:  # 회전이 필요한 경우
            #         angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            #         # 각도를 도 단위로 변환하고 제한 (-15도 ~ +15도)
            #         angle_deg = np.degrees(angle)
            #         angle_deg = np.clip(angle_deg, -15, 15)
                    
            #         # 회전 적용
            #         pixels_2d = ndimage.rotate(pixels_2d, -angle_deg, reshape=False, mode='constant', cval=0)
            
            pixels = pixels_2d.flatten()
            
            # 5. 자동 반전 판단: 평균 픽셀 값으로 배경 색상 판단
            mean_pixel = np.mean(pixels)
            if mean_pixel > 0.5:
                # 평균이 0.5보다 크면 흰 배경 → 반전 필요
                pixels = 1.0 - pixels
                st.info("💡 흰 배경 감지: 이미지를 반전하여 MNIST 형식으로 변환했습니다.")
            else:
                # 평균이 0.5보다 작으면 검은 배경 → 반전 불필요
                st.info("💡 검은 배경 감지: 이미지가 이미 MNIST 형식입니다.")
            
            # NaN 값 처리 및 범위 보정
            pixels = np.array(pixels)
            pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)  # NaN을 0으로, 무한대를 0/1로 변환
            pixels = np.clip(pixels, 0.0, 1.0)  # 0-1 범위로 클리핑
            
            # 리스트로 변환
            pixels = pixels.tolist()
            
            # 디버깅: 픽셀 값 범위 확인
            st.info(f"고급 전처리 완료: {min(pixels):.3f} ~ {max(pixels):.3f}")
            st.info(f"픽셀 개수: {len(pixels)}, NaN 개수: {sum(1 for p in pixels if np.isnan(p))}")
            st.info("💡 가우시안 블러 + Otsu 이진화 + 중앙정렬 + 회전보정을 적용했습니다.")
            
        except Exception as e:
            st.warning(f"전처리 중 오류 발생: {str(e)}")
            st.info("기본 전처리로 진행합니다...")
            
            # 기본 전처리 (fallback)
            pixels = np.array(image).flatten()
            pixels = pixels / 255.0  # 정규화
            
            # 자동 반전 판단
            mean_pixel = np.mean(pixels)
            if mean_pixel > 0.5:
                pixels = 1.0 - pixels
                st.info("💡 흰 배경 감지: 이미지를 반전하여 MNIST 형식으로 변환했습니다.")
            else:
                st.info("💡 검은 배경 감지: 이미지가 이미 MNIST 형식입니다.")
            
            # NaN 값 처리 및 범위 보정
            pixels = np.array(pixels)
            pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
            pixels = np.clip(pixels, 0.0, 1.0)
            
            pixels = pixels.tolist()
            
            st.info(f"기본 전처리 완료: {min(pixels):.3f} ~ {max(pixels):.3f}")
        
        # 픽셀 값 시각화
        st.subheader("📊 픽셀 값 분포")
        st.bar_chart(pixels[:784])  # 첫 50개 픽셀만 표시
        
        # 예측 버튼
        if st.button("🔍 숫자 예측하기", type="primary"):
            with st.spinner("AI가 숫자를 분석 중..."):
                try:
                    # 선택된 모델에 따라 적절한 API 엔드포인트 사용
                    selected_api_url = get_api_url(model_choice)
                    data = {"pixels": pixels}
                    response = requests.post(selected_api_url, json=data)
                    
                    prediction = None
                    confidence = None
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['prediction']
                        confidence = result['confidence']
                    else:
                        st.error(f"API 오류: {response.status_code}")
                        st.error(response.text)
                    
                    # 예측이 성공한 경우에만 결과 표시
                    if prediction is not None and confidence is not None:
                        with col2:
                            st.header("🎯 예측 결과")
                            
                            # 큰 숫자로 예측 결과 표시
                            st.markdown(f"""
                            <div style="text-align: center; padding: 2rem;">
                                <h1 style="font-size: 8rem; margin: 0; color: #1f77b4;">{prediction}</h1>
                                <p style="font-size: 1.5rem; color: #666;">예측된 숫자</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # 신뢰도 표시
                            st.metric("신뢰도", f"{confidence:.2%}")
                            
                            # 사용된 모델 표시
                            st.info(f"🤖 사용된 모델: **{model_choice}**")
                            
                            # 신뢰도 바
                            st.progress(confidence)
                            
                            # 결과 설명
                            if confidence > 0.8:
                                st.success("🎉 높은 신뢰도로 예측되었습니다!")
                            elif confidence > 0.5:
                                st.warning("⚠️ 보통 신뢰도로 예측되었습니다.")
                            else:
                                st.error("❌ 낮은 신뢰도입니다. 다른 이미지를 시도해보세요.")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚫 FastAPI 서버에 연결할 수 없습니다.")
                    st.info("서버가 실행 중인지 확인해주세요: `uvicorn src.main:app --reload`")
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")

with col2:
    if uploaded_file is None:
        st.header("📋 사용 방법")
        st.markdown("""
        1. **이미지 업로드**: 왼쪽에서 숫자 이미지를 업로드하세요
        2. **자동 전처리**: 이미지가 28x28 그레이스케일로 변환됩니다
        3. **예측 실행**: "숫자 예측하기" 버튼을 클릭하세요
        4. **결과 확인**: AI가 예측한 숫자와 신뢰도를 확인하세요
        """)
        
        st.header("💡 팁")
        st.markdown("""
        - **이미지 크기**: 28x28 픽셀에 가까운 이미지가 좋습니다
        - **배경**: 흰 배경에 검은 숫자도 자동으로 변환됩니다
        - **선명도**: 선명하고 명확한 숫자 이미지를 사용하세요
        - **자동 변환**: 이미지가 MNIST 형식으로 자동 변환됩니다
        """)

# 푸터
st.markdown("---")
st.markdown("🤖 Powered by XGBoost + FastAPI + Streamlit")
