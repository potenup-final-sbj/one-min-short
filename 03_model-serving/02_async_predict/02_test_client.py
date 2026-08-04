import asyncio
import httpx
import time
"""
uv add httpx
"""

async def call_fast_inference(client):
    print("1. [fast inference] 요청 시작...")
    start_time = time.perf_counter()

    response = await client.get("/fast-inference", params={"text": "This is a great movie!"})

    end_time = time.perf_counter()
    print(f"1. [fast inference] 응답 도착! 소요 시간: {end_time - start_time:.2f}초")
    return response.json()

async def call_ping(client):
    # Fast Inference가 시작된 후 아주 짧게(0.5초) 기다렸다가 보냅니다.
    # 기다리는 이유는 Fast Inference가 실행 중일 때 Ping 요청을 보내기 위함입니다.
    await asyncio.sleep(0.5)
    print("2. [Ping] 요청 시작 (루프가 살아있다면 즉시 응답해야 함)...")
    
    start_time = time.perf_counter()
    
    response = await client.get("/ping")
    
    end_time = time.perf_counter()
    
    print(f"2. [Ping] 응답 도착! 소요 시간: {end_time - start_time:.2f}초")
    return response.json()


async def call_blocking(client):
    # Fast Inference가 시작된 후 아주 짧게(0.5초) 기다렸다가 보냅니다.
    # 기다리는 이유는 Fast Inference가 실행 중일 때 Blocking 요청을 보내기 위함입니다.
    await asyncio.sleep(0.5)
    print("3. [Blocking] 요청 시작 (서버가 블로킹 된다면 응답이 늦어짐)...")
    
    start_time = time.perf_counter()
    
    response = await client.get("/blocking")
    
    end_time = time.perf_counter()
    
    print(f"3. [Blocking] 응답 도착! 소요 시간: {end_time - start_time:.2f}초")
    return response.json()



async def main():
    for i in range(3):
        print(f"\n=== 테스트 라운드 {i + 1} 시작 ===")
        # 서버 주소를 확인하세요 (기본 8000포트)
        # httpx.AsyncClient를 사용하여 비동기 HTTP 클라이언트를 생성합니다.
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
            # 두 요청을 동시에 실행합니다.
            await asyncio.gather(
                call_fast_inference(client),
                call_ping(client),
                call_blocking(client)
            )

if __name__ == "__main__":
    asyncio.run(main())
    
    
"""
fast-inference 요청이 진행 중일 때도 ping 요청이 즉시 응답하는 것을 관찰할 수 있습니다.
이는 fast-inference가 별도의 스레드에서 실행되기 때문에 이벤트 루프가 차단되지 않기 때문입니다.


=== 테스트 라운드 1 시작 ===
1. [fast inference] 요청 시작...
2. [Ping] 요청 시작 (루프가 살아있다면 즉시 응답해야 함)...
2. [Ping] 응답 도착! 소요 시간: 0.00초
1. [fast inference] 응답 도착! 소요 시간: 5.05초
"""