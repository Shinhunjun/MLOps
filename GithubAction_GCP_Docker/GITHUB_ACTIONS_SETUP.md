# GitHub Actions Setup Guide

## Required GitHub Secrets

GitHub Actions 워크플로우를 실행하려면 다음 Secret을 설정해야 합니다:

### 1. GCP_SA_KEY

**생성 방법:**
1. 이미 생성한 `mlops-compute-lab-3cd8e7632150.json` 파일 사용
2. 파일 전체 내용을 복사

**GitHub에 추가:**
1. Repository → Settings → Secrets and variables → Actions
2. "New repository secret" 클릭
3. Name: `GCP_SA_KEY`
4. Value: JSON 파일 전체 내용 붙여넣기
5. "Add secret" 클릭

## Workflow 트리거 방법

### 1. 자동 트리거 (FastAPI에서)

FastAPI에서 10개 피드백 데이터가 모이면 자동으로 트리거됩니다:

```python
# FastAPI에서 실행
POST https://api.github.com/repos/Shinhunjun/MLOps/dispatches
Headers:
  Authorization: token <GITHUB_TOKEN>
Body:
  {
    "event_type": "retrain-model",
    "client_payload": {
      "dataset_version": "sub_set_5"
    }
  }
```

### 2. 수동 트리거 (GitHub UI에서)

1. GitHub Repository → Actions 탭
2. "Retrain MNIST Model" 워크플로우 선택
3. "Run workflow" 버튼 클릭
4. Subset ID 입력 (예: `sub_set_0`)
5. Epochs 입력 (기본값: 5)
6. "Run workflow" 실행

## Workflow 동작 과정

1. **데이터 다운로드**: GCS에서 피드백 데이터 다운로드
2. **모델 재학습**: 새 데이터와 MNIST 데이터 결합하여 학습
3. **모델 평가**: 테스트 데이터로 정확도 측정
4. **Vertex AI 업로드**: 학습된 모델을 Vertex AI Model Registry에 등록
5. **메타데이터 저장**: 학습 결과를 GitHub Artifacts에 저장

## 확인 방법

### 1. 워크플로우 실행 상태
- GitHub → Actions → 해당 워크플로우 클릭
- 각 Step별 로그 확인 가능

### 2. 학습 메타데이터
- 워크플로우 완료 후 Artifacts 다운로드
- `metadata.json` 파일에서 정확도, 손실 등 확인

### 3. Vertex AI에서 확인
- GCP Console → Vertex AI → Model Registry
- 새로운 모델 버전 확인

## 문제 해결

### Secret이 없을 때
```
Error: google-github-actions/auth@v2 failed with: missing required field 'credentials_json'
```
→ `GCP_SA_KEY` Secret 추가

### 권한 오류
```
Error: Permission denied on resource 'projects/mlops-compute-lab/...'
```
→ 서비스 계정에 필요한 권한 추가:
- Vertex AI User
- Storage Object Admin

### 데이터가 없을 때
```
Error: Downloaded 0 new samples
```
→ GCS에서 해당 subset 데이터 확인
