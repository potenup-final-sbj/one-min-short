import asyncio
import httpx
import time
"""
uv add httpx
"""

async def call_slow_inference(client):
    print("1. [Slow Inference] 요청 시작...")
    start_time = time.perf_counter()

    response = await client.get("/slow-inference", params={"text": "This is a great movie!"})

    end_time = time.perf_counter()
    print(f"1. [Slow Inference] 응답 도착! 소요 시간: {end_time - start_time:.2f}초")
    return response.json()

async def call_ping(client):
    # Slow Inference가 시작된 후 아주 짧게(0.5초) 기다렸다가 보냅니다.
    # 기다리는 이유는 Slow Inference가 실행 중일 때 Ping 요청을 보내기 위함입니다.
    await asyncio.sleep(0.5)
    print("2. [Ping] 요청 시작 (루프가 살아있다면 즉시 응답해야 함)...")
    
    start_time = time.perf_counter()
    
    response = await client.get("/ping")
    
    end_time = time.perf_counter()
    
    print(f"2. [Ping] 응답 도착! 소요 시간: {end_time - start_time:.2f}초")
    return response.json()

async def main():
    for i in range(3):
        print(f"\n=== 테스트 라운드 {i + 1} 시작 ===")
        # 서버 주소를 확인하세요 (기본 8000포트)
        # httpx.AsyncClient를 사용하여 비동기 HTTP 클라이언트를 생성합니다.
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
            # 두 요청을 동시에 실행합니다.
            await asyncio.gather(
                call_slow_inference(client),
                call_ping(client)
            )

if __name__ == "__main__":
    asyncio.run(main())
    
    
"""
ping 요청은 즉시 응답해야 하지만, slow-inference 요청이 진행 중일 때는 응답하지 못하는 현상을 관찰할 수 있습니다.
이는 slow-inference가 이벤트 루프를 점유하고 있기 때문으로, 비동기 서버 환경에서 CPU-bound 작업을 처리할 때 발생할 수 있는 문제점입니다.

=== 테스트 라운드 1 시작 ===
1. [Slow Inference] 요청 시작...
2. [Ping] 요청 시작 (루프가 살아있다면 즉시 응답해야 함)...
1. [Slow Inference] 응답 도착! 소요 시간: 7.83초
2. [Ping] 응답 도착! 소요 시간: 7.32초
"""