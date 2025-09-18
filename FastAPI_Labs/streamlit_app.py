import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import os
import time
import glob
import subprocess

def count_new_data():
    """new_data 폴더의 파일 개수를 세어 반환"""
    data_dir = "new_data"
    if not os.path.exists(data_dir):
        return 0
    
    # 이미지 파일들만 카운트
    image_files = glob.glob(os.path.join(data_dir, "*.png"))
    image_files.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
    
    return len(image_files)

def check_and_trigger_retrain():
    """데이터 개수를 체크하고 10개 이상이면 자동으로 재훈련 트리거"""
    data_count = count_new_data()
    
    if data_count >= 10:
        st.success(f"🎯 {data_count}개의 데이터가 수집되었습니다!")
        st.info("🚀 GitHub Actions에서 모델 재훈련을 시작합니다...")
        
        try:
            # trigger_retrain.py 실행 (GitHub Actions 트리거만)
            result = subprocess.run(
                ["python", "trigger_retrain.py"], 
                capture_output=True, 
                text=True,
                cwd="."
            )
            
            if result.returncode == 0:
                st.success("✅ 재훈련 트리거가 성공적으로 실행되었습니다!")
                st.info("GitHub Actions에서 모델 재훈련이 진행됩니다.")
                st.info("재훈련 완료 후 자동으로 모델이 업데이트됩니다.")
                
                # 트리거 성공 후 로컬 데이터 삭제 (충돌 방지)
                clear_local_data()
                
            else:
                st.error(f"❌ 트리거 실행 실패: {result.stderr}")
                
        except Exception as e:
            st.error(f"❌ 트리거 실행 중 오류: {e}")
    else:
        st.info(f"📊 현재 데이터: {data_count}개 (10개까지 {10-data_count}개 더 필요)")

def clear_local_data():
    """로컬 new_data 폴더를 삭제하여 충돌 방지"""
    import shutil
    data_dir = "new_data"
    
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
            print("🗑️ 로컬 new_data 폴더가 삭제되었습니다. (충돌 방지)")
            st.info("🗑️ 로컬 데이터가 삭제되었습니다. (충돌 방지)")
        except Exception as e:
            print(f"⚠️ 데이터 삭제 실패: {e}")
    else:
        print("📝 삭제할 데이터가 없습니다.")

# 페이지 설정
st.set_page_config(
    page_title="MNIST 숫자 인식기",
    page_icon="🔢",
    layout="wide"
)

# --- State Management ---
# Initialize session state keys
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'feedback_mode' not in st.session_state:
    st.session_state.feedback_mode = False
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

# 제목
st.title("🔢 MNIST 숫자 인식기")
st.markdown("이미지를 업로드하면 AI가 숫자를 예측해드립니다!")

# 사이드바
st.sidebar.header("설정")

api_url = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8000/predict",
    help="FastAPI CNN 서버의 엔드포인트 URL"
)

st.sidebar.markdown("---")
st.sidebar.header("🔄 모델 관리")

if st.sidebar.button("🔄 모델 자동 업데이트", use_container_width=True):
    with st.spinner("최신 모델을 가져오는 중..."):
        try:
            response = requests.post("http://localhost:8000/auto-update-model", timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.sidebar.success(result["message"])
                st.rerun()
            else:
                st.sidebar.error(f"업데이트 실패: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"업데이트 중 오류: {e}")

if st.sidebar.button("🔄 모델 리로드", use_container_width=True):
    with st.spinner("모델을 리로드하는 중..."):
        try:
            response = requests.post("http://localhost:8000/reload-models", timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(f"리로드 실패: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"리로드 중 오류: {e}")


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
        # Check if this is a new file. If so, clear old state.
        if st.session_state.current_file_id != f"{uploaded_file.name}-{uploaded_file.size}":
            st.session_state.current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.feedback_submitted = False
            st.session_state.feedback_mode = False

        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        # 이미지 전처리 (백그라운드에서 실행)
        # 그레이스케일 변환
        if image.mode != 'L':
            image = image.convert('L')
        
        # 28x28 크기로 리사이즈
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 픽셀 값 추출 및 전처리
        try:
            pixels = np.array(image).flatten()
            pixels_2d = pixels.reshape(28, 28)
            pixels_2d = pixels_2d / 255.0 if pixels_2d.max() > 1.0 else pixels_2d
            pixels = pixels_2d.flatten()
            
            mean_pixel = np.mean(pixels)
            if mean_pixel > 0.5:
                pixels = 1.0 - pixels
            
            pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
            pixels = np.clip(pixels, 0.0, 1.0)
            pixels = pixels.tolist()
            
        except Exception as e:
            st.error(f"전처리 중 오류 발생: {str(e)}")
            pixels = None

        if pixels:
            
            # 예측 버튼
            if st.button("🔍 숫자 예측하기", type="primary"):
                with st.spinner("AI가 숫자를 분석 중..."):
                    try:
                        data = {"pixels": pixels}
                        response = requests.post(api_url, json=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.prediction = result['prediction']
                            st.session_state.confidence = result['confidence']
                            # Clear previous feedback state for the new prediction
                            st.session_state.feedback_submitted = False
                            st.session_state.feedback_mode = False
                            
                            # 자동 트리거 체크 (예측 후)
                            check_and_trigger_retrain()
                        else:
                            st.error(f"API 오류: {response.status_code}")
                            st.error(response.text)
                            st.session_state.prediction = None
                    
                    except requests.exceptions.ConnectionError:
                        st.error("🚫 FastAPI 서버에 연결할 수 없습니다.")
                        st.info("서버가 실행 중인지 확인해주세요: `uvicorn src.main:app --reload`")
                        st.session_state.prediction = None
                    except Exception as e:
                        st.error(f"오류가 발생했습니다: {str(e)}")
                        st.session_state.prediction = None

# Display results and feedback if a prediction exists in the session state
if st.session_state.prediction is not None:
    with col2:
        st.header("🎯 예측 결과")
        
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 8rem; margin: 0; color: #1f77b4;">{prediction}</h1>
            <p style="font-size: 1.5rem; color: #666;">예측된 숫자</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("신뢰도", f"{confidence:.2%}")
        
        # 모델 정보 표시
        try:
            import requests
            response = requests.get("http://localhost:8000/model-info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                st.info(f"🤖 사용된 모델: **{model_info.get('model_name', 'CNN (FastAPI)')}**")
            else:
                st.info(f"🤖 사용된 모델: **CNN (FastAPI)**")
        except:
            st.info(f"🤖 사용된 모델: **CNN (FastAPI)**")
        
        st.progress(confidence)
        
        if confidence > 0.8:
            st.success("🎉 높은 신뢰도로 예측되었습니다!")
        elif confidence > 0.5:
            st.warning("⚠️ 보통 신뢰도로 예측되었습니다.")
        else:
            st.error("❌ 낮은 신뢰도입니다. 다른 이미지를 시도해보세요.")

        # --- 피드백 섹션 ---
        st.markdown("---")
        st.subheader("🤔 이 예측을 데이터셋에 추가할까요?")

        if not st.session_state.feedback_submitted:
            col_feedback_1, col_feedback_2 = st.columns(2)

            if col_feedback_1.button(f"👍 네, '{prediction}'이 맞습니다."):
                try:
                    # FastAPI에 피드백 데이터 전송
                    feedback_data = {
                        "pixels": pixels,
                        "label": prediction
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/save-feedback",
                        json=feedback_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result["message"])
                        
                        if result.get("triggered"):
                            st.info("🚀 GitHub Actions에서 모델 재훈련이 진행됩니다!")
                        
                        st.session_state.feedback_submitted = True
                        st.rerun()
                    else:
                        st.error(f"피드백 저장 실패: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"피드백 저장 중 오류 발생: {e}")

            if col_feedback_2.button("👎 아니요, 틀렸습니다."):
                st.session_state.feedback_submitted = True
                st.session_state.feedback_mode = True
                st.rerun()

        elif st.session_state.feedback_mode:
            correct_label = st.text_input("정확한 숫자를 입력해주세요:")
            if st.button("피드백 제출"):
                if correct_label.isdigit() and 0 <= int(correct_label) <= 9:
                    try:
                        # FastAPI에 피드백 데이터 전송
                        feedback_data = {
                            "pixels": pixels,
                            "label": int(correct_label)
                        }
                        
                        response = requests.post(
                            "http://localhost:8000/save-feedback",
                            json=feedback_data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(result["message"])
                            
                            if result.get("triggered"):
                                st.info("🚀 GitHub Actions에서 모델 재훈련이 진행됩니다!")
                            
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_mode = False
                            st.rerun()
                        else:
                            st.error(f"피드백 저장 실패: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"피드백 저장 중 오류 발생: {e}")
                else:
                    st.warning("0에서 9 사이의 숫자를 입력해주세요.")
        
        else: # Feedback has been submitted
            st.info("피드백이 기록되었습니다. 감사합니다!")

# Display instructions if no file is uploaded
if uploaded_file is None:
    with col2:
        st.header("📋 사용 방법")
        st.markdown("""
        1. **이미지 업로드**: 왼쪽에서 숫자 이미지를 업로드하세요
        2. **예측 실행**: "숫자 예측하기" 버튼을 클릭하세요
        3. **결과 확인**: AI가 예측한 숫자와 신뢰도를 확인하세요
        4. **피드백 제공**: 예측이 맞는지 피드백을 제공하세요
        """)
        
        st.header("💡 팁")
        st.markdown("""
        - **이미지 크기**: 28x28 픽셀에 가까운 이미지가 좋습니다
        - **배경**: 흰 배경에 검은 숫자도 자동으로 변환됩니다
        - **선명도**: 선명하고 명확한 숫자 이미지를 사용하세요
        - **피드백**: 정확한 피드백을 제공하면 모델이 더 정확해집니다
        """)

# 푸터
st.markdown("---")
st.markdown("🤖 Powered by CNN + FastAPI + Streamlit")