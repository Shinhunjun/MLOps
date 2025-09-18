#!/usr/bin/env python3
"""
데이터가 100개 모이면 GitHub Actions를 트리거하는 스크립트
"""
import os
import json
import requests
import glob
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def count_new_data():
    """new_data 폴더의 sub_set 파일 개수를 세어 반환"""
    data_dir = "new_data"
    if not os.path.exists(data_dir):
        return 0
    
    # count.json 파일 확인
    count_file = os.path.join(data_dir, "count.json")
    if not os.path.exists(count_file):
        return 0
    
    with open(count_file, 'r') as f:
        count_data = json.load(f)
    
    # 현재 sub_set 폴더 확인
    current_sub_set = f"sub_set_{count_data['sub_set_count'] - 1}"  # 이전 sub_set
    sub_set_path = os.path.join(data_dir, current_sub_set)
    
    if not os.path.exists(sub_set_path):
        return 0
    
    # 이미지 파일들만 카운트
    image_files = glob.glob(os.path.join(sub_set_path, "*.png"))
    image_files.extend(glob.glob(os.path.join(sub_set_path, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(sub_set_path, "*.jpeg")))
    
    return len(image_files)

def commit_and_push_data():
    """데이터를 Git에 커밋하고 푸시"""
    try:
        import subprocess
        
        # Git 상태 확인
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='.')
        
        if not result.stdout.strip():
            print("📝 커밋할 변경사항이 없습니다.")
            return True
        
        # 모든 변경사항 추가
        subprocess.run(['git', 'add', '.'], check=True, cwd='.')
        
        # 커밋
        commit_message = f"feat: Add {count_new_data()} new training data points"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, cwd='.')
        
        # 푸시
        subprocess.run(['git', 'push'], check=True, cwd='.')
        
        print("✅ 데이터가 성공적으로 Git에 푸시되었습니다!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git 작업 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def trigger_github_action():
    """GitHub repository dispatch를 통해 워크플로우 트리거"""
    # GitHub Personal Access Token이 필요합니다
    # Settings > Developer settings > Personal access tokens > Generate new token
    # 권한: repo, workflow
    
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN 환경변수가 설정되지 않았습니다.")
        print("GitHub Personal Access Token을 설정해주세요:")
        print("export GITHUB_TOKEN='your_token_here'")
        return False
    
    # 1. 먼저 데이터를 Git에 커밋하고 푸시
    if not commit_and_push_data():
        print("❌ 데이터 푸시에 실패했습니다.")
        return False
    
    # 2. GitHub API를 통해 repository dispatch 트리거
    url = "https://api.github.com/repos/{owner}/{repo}/dispatches"
    
    # .env 파일에서 GitHub 정보 로드
    owner = os.getenv('GITHUB_OWNER', 'hunjunsin')
    repo = os.getenv('GITHUB_REPO', 'MLOps')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "event_type": "retrain-model",
        "client_payload": {
            "trigger_reason": "Data collection threshold reached",
            "data_count": str(count_new_data()),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    try:
        response = requests.post(url.format(owner=owner, repo=repo), 
                               headers=headers, 
                               json=data)
        
        if response.status_code == 204:
            print("✅ GitHub Actions 워크플로우가 성공적으로 트리거되었습니다!")
            return True
        else:
            print(f"❌ 워크플로우 트리거 실패: {response.status_code}")
            print(f"응답: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    """메인 함수"""
    print("🔍 새로운 데이터 개수 확인 중...")
    
    data_count = count_new_data()
    print(f"📊 현재 new_data 폴더의 파일 개수: {data_count}")
    
    if data_count >= 10:  # 테스트를 위해 10개로 변경
        print("🎯 10개 이상의 데이터가 수집되었습니다!")
        print("🚀 GitHub Actions 워크플로우를 트리거합니다...")
        
        if trigger_github_action():
            print("✅ 재훈련 프로세스가 시작되었습니다.")
            print("GitHub Actions 탭에서 진행상황을 확인할 수 있습니다.")
        else:
            print("❌ 워크플로우 트리거에 실패했습니다.")
    else:
        print(f"⏳ 아직 {10 - data_count}개 더 필요합니다.")
        print("더 많은 데이터를 수집한 후 다시 실행해주세요.")

if __name__ == "__main__":
    main()
