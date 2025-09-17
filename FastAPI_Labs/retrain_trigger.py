import os
import subprocess
import sys
import shutil

# --- Configuration ---
# The script is in FastAPI_Labs, so paths are relative to that
NEW_DATA_DIR = 'new_data'
ARCHIVE_DIR = 'archived_data'
TRAINING_SCRIPT_PATH = os.path.join('src', 'train_cnn.py')
RETRAIN_THRESHOLD = 10 # As requested by the user

def run_retraining():
    """
    Executes the retraining script as a subprocess.
    """
    print(f"--- '{TRAINING_SCRIPT_PATH}' 실행 ---")
    try:
        # We run the script using python -u for unbuffered output
        process = subprocess.run(
            [sys.executable, '-u', TRAINING_SCRIPT_PATH],
            check=True,
            text=True,
            capture_output=True # Capture stdout and stderr
        )
        print(process.stdout)
        if process.stderr:
            print("--- Stderr: ---")
            print(process.stderr)
        print("--- 훈련 스크립트 성공적으로 실행됨 ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! 훈련 스크립트 실행 중 오류 발생 !!!")
        print(f"Return code: {e.returncode}")
        print("--- Stdout: ---")
        print(e.stdout)
        print("--- Stderr: ---")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(f"!!! 오류: '{TRAINING_SCRIPT_PATH}'을(를) 찾을 수 없습니다. 경로를 확인하세요.")
        return False

def archive_processed_data(files_to_archive):
    """
    Moves a list of files from the new_data directory to the archive directory.
    """
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
        print(f"'{ARCHIVE_DIR}' 디렉토리가 생성되었습니다.")

    print(f"{len(files_to_archive)}개의 파일을 '{ARCHIVE_DIR}'(으)로 이동합니다...")
    for filename in files_to_archive:
        src_path = os.path.join(NEW_DATA_DIR, filename)
        dest_path = os.path.join(ARCHIVE_DIR, filename)
        try:
            shutil.move(src_path, dest_path)
        except Exception as e:
            print(f"'{filename}' 파일 이동 중 오류 발생: {e}")
    print("--- 데이터 보관 완료 ---")


def main():
    """
    Main function to check for new data and trigger retraining.
    """
    print("=" * 60)
    print("재훈련 트리거 스크립트 시작")
    print("=" * 60)

    if not os.path.exists(NEW_DATA_DIR):
        print(f"'{NEW_DATA_DIR}' 디렉토리가 없습니다. 스크립트를 종료합니다.")
        return

    image_files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith('.png')]
    file_count = len(image_files)

    print(f"현재 수집된 데이터: {file_count}개")
    print(f"재훈련 필요 데이터 수: {RETRAIN_THRESHOLD}개")

    if file_count >= RETRAIN_THRESHOLD:
        print(f"\n재훈련 임계점에 도달했습니다. ({file_count} >= {RETRAIN_THRESHOLD})")
        print("재훈련을 시작합니다...")
        
        # Run the training script
        success = run_retraining()
        
        if success:
            # If training was successful, archive the files that were used
            print("\n훈련이 성공적으로 완료되었습니다. 처리된 데이터를 보관합니다.")
            archive_processed_data(image_files)
        else:
            print("\n!!! 훈련 스크립트 실패. 데이터는 보관되지 않았습니다. 오류를 확인하세요. !!!")

    else:
        needed = RETRAIN_THRESHOLD - file_count
        print(f"\n재훈련을 시작하려면 {needed}개의 데이터가 더 필요합니다.")

    print("\n" + "=" * 60)
    print("스크립트 실행 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
