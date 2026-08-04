"""
허깅페이스 허브에 저장된 모델 캐시를 스캔하는 도구입니다.
허깅페이스의 blob 캐시를 통해 전역에서 다운로드된 모델 파일들을 확인할 수 있으며,
개별 프로젝트는 이 캐시를 참조하여 모델을 메모리에 올리는 방식으로 동작합니다.
이를 통해 프로젝트는 매번 모델을 새로 다운로드하지 않고도 빠르게 로드할 수 있으며,
디스크 공간을 효율적으로 사용할 수 있습니다.
"""

import os

from huggingface_hub import scan_cache_dir


def analyze_hf_cache():
    """
    현재 시스템에 저장된 Hugging Face 모델 캐시를 스캔합니다.
    OS 레벨에서 물리적 용량을 얼마나 차지하는지 확인하는 시니어의 도구입니다.
    """
    cache_info = scan_cache_dir()
    print("--- HF 캐시 리포트 ---")
    for repo in cache_info.repos:
        repo_id = repo.repo_id
        size_on_disk = repo.size_on_disk_str
        print(f"모델: {repo_id} | 용량: {size_on_disk}")

    # 캐시 폴더 위치 출력 (OS 환경마다 다름)
    print(f"\n캐시 경로: {os.path.expanduser('~/.cache/huggingface/hub')}")


if __name__ == "__main__":
    # 이 코드를 실행하려면 pip install huggingface_hub 가 필요합니다.
    analyze_hf_cache()
