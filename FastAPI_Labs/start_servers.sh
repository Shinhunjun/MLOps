#!/bin/bash

echo "🚀 MLOps 서버들을 시작합니다..."

# 1. FastAPI 서버 시작 (백그라운드)
echo "📡 FastAPI 서버 시작 중..."
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# 2. 자동 업데이터 시작 (백그라운드)
echo "🔄 자동 모델 업데이터 시작 중..."
cd ..
python auto_model_updater.py &
UPDATER_PID=$!

# 3. Streamlit 서버 시작 (포그라운드)
echo "🌐 Streamlit 서버 시작 중..."
streamlit run streamlit_app.py --server.port 8501

# 종료 시 모든 프로세스 정리
cleanup() {
    echo "🛑 서버들을 종료합니다..."
    kill $FASTAPI_PID 2>/dev/null
    kill $UPDATER_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM
