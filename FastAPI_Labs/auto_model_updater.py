#!/usr/bin/env python3
"""
주기적으로 모델을 업데이트하는 스크립트
10분마다 Git pull을 실행하고 모델을 리로드합니다.
"""
import time
import subprocess
import requests
import os
from datetime import datetime

def git_pull_and_reload():
    """Git pull을 실행하고 FastAPI에 리로드 요청을 보냅니다."""
    try:
        print(f"[{datetime.now()}] 🔄 모델 업데이트 체크 중...")
        
        # 1. Git pull 실행
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            if "Already up to date" in result.stdout:
                print("📝 이미 최신 버전입니다.")
                return False
            else:
                print("✅ 새로운 변경사항을 가져왔습니다.")
                
                # 2. FastAPI에 리로드 요청
                try:
                    response = requests.post("http://localhost:8000/reload-models", timeout=10)
                    if response.status_code == 200:
                        print("🔄 모델이 성공적으로 리로드되었습니다.")
                        return True
                    else:
                        print(f"❌ 모델 리로드 실패: {response.status_code}")
                        return False
                except requests.exceptions.ConnectionError:
                    print("⚠️ FastAPI 서버에 연결할 수 없습니다.")
                    return False
        else:
            print(f"❌ Git pull 실패: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 업데이트 중 오류 발생: {e}")
        return False

def main():
    """메인 함수 - 10분마다 업데이트 체크"""
    print("🚀 자동 모델 업데이터를 시작합니다...")
    print("⏰ 10분마다 모델을 업데이트합니다.")
    print("🛑 종료하려면 Ctrl+C를 누르세요.")
    
    update_count = 0
    
    try:
        while True:
            # 10분 대기
            time.sleep(600)  # 600초 = 10분
            
            # 업데이트 실행
            if git_pull_and_reload():
                update_count += 1
                print(f"📊 총 {update_count}번의 업데이트가 완료되었습니다.")
            
    except KeyboardInterrupt:
        print("\n🛑 자동 업데이터를 종료합니다.")
        print(f"📊 총 {update_count}번의 업데이트가 완료되었습니다.")

if __name__ == "__main__":
    main()
