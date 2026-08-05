from __future__ import annotations

import re


GENRE_STYLES = {
    "로맨스": {"accent": "#ff4d8d", "location": "도심 오피스"},
    "스릴러": {"accent": "#8b5cf6", "location": "불 꺼진 사무실"},
    "코미디": {"accent": "#ffb703", "location": "활기찬 스타트업"},
    "판타지": {"accent": "#22d3ee", "location": "시간이 멈춘 빌딩"},
}


def _title_from_prompt(prompt: str) -> str:
    cleaned = re.sub(r"[.!?。]+$", "", prompt.strip())
    return cleaned if len(cleaned) <= 28 else cleaned[:27].rstrip() + "…"


def _scene(
    scene_id: str,
    phase: str,
    title: str,
    location: str,
    speaker: str,
    dialogue: str,
    visual: str,
    motion_prompt: str,
    duration: int,
    accent: str,
) -> dict:
    return {
        "id": scene_id,
        "phase": phase,
        "title": title,
        "location": location,
        "speaker": speaker,
        "dialogue": dialogue,
        "subtitle": dialogue,
        "visual_prompt": visual,
        "motion_prompt": motion_prompt,
        "duration": duration,
        "accent": accent,
    }


def create_story(prompt: str, genre: str, mood: str) -> dict:
    style = GENRE_STYLES.get(genre, GENRE_STYLES["로맨스"])
    accent = style["accent"]
    location = style["location"]
    title = _title_from_prompt(prompt)

    common = [
        _scene(
            "common_01", "common", "모든 것이 시작된 날", location,
            "내레이션", f"{prompt}. 모든 것은 바로 그날 시작됐다.",
            f"{mood} 분위기, 세로형 드라마 오프닝, {location}",
            "The woman takes a slow breath and looks around the office. Her hair and blazer move naturally. Coworkers work softly in the background. Slow cinematic push-in.",
            8, accent,
        ),
        _scene(
            "common_02", "common", "낯선 신호", location,
            "서윤", "설마… 이게 정말 우연이라고? 심장이 왜 이렇게 뛰지?",
            "놀란 주인공의 클로즈업, 흔들리는 눈빛",
            "She slowly raises her gaze, blinks, and freezes in recognition. Her expression changes from calm to shocked. Subtle handheld camera movement.",
            8, accent,
        ),
        _scene(
            "common_03", "common", "피할 수 없는 대면", location,
            "도현", "오랜만이네. 나한테 정말 아무 말도 없었어?",
            "차가운 표정의 상대역, 역광, 긴장감 있는 재회",
            "The man walks into the office, then turns his head toward the woman. They lock eyes. Employees applaud naturally behind them. Slow camera dolly forward.",
            8, accent,
        ),
        _scene(
            "common_04", "common", "숨겨진 관계", location,
            "민지", "잠깐만요. 두 사람… 원래 아는 사이예요?",
            "세 사람 사이의 불편한 침묵, 시선 교차",
            "The two leads remain tense while a coworker glances between them in confusion. Natural blinking and breathing. The camera gently shifts focus between their faces.",
            8, accent,
        ),
        _scene(
            "common_05", "common", "당신의 선택은?", "선택의 순간",
            "내레이션", "모른 척 외면할까, 아니면 그날의 진실을 물을까?",
            "시간이 멈춘 듯한 주인공, 두 갈래 선택",
            "The woman hesitates and looks away, then slowly turns back toward the man. The camera circles subtly as the office background falls out of focus.",
            8, "#f43f5e",
        ),
    ]

    ending_a = [
        _scene(
            "ending_a_01", "ending_a", "A. 모른 척한다", location,
            "서윤", "죄송하지만… 저희 오늘 처음 뵙는 사이 아닌가요?",
            "감정을 숨기고 돌아서는 주인공",
            "She hides her emotion, gives a restrained professional nod, and slowly turns away. The man watches without moving. Smooth cinematic tracking shot.",
            10, "#3b82f6",
        ),
        _scene(
            "ending_a_02", "ending_a", "끝나지 않은 기억", "대표실 앞",
            "도현", "그래, 그렇게 하자. 하지만 네가 먼저 날 찾아오게 될 거야.",
            "의미심장하게 미소 짓는 상대역, 다음 화 예고",
            "The man watches her leave, lowers his eyes briefly, then gives a subtle knowing smile. Slow close-up with natural facial movement.",
            10, "#3b82f6",
        ),
    ]

    ending_b = [
        _scene(
            "ending_b_01", "ending_b", "B. 진실을 묻는다", location,
            "서윤", "8년 전, 왜 아무 말도 없이 사라졌어? 지금 대답해.",
            "눈물을 참으며 정면으로 맞서는 주인공",
            "She steps closer and confronts him, eyes trembling with restrained tears. He stiffens and meets her gaze. Slow dramatic push-in.",
            10, "#f59e0b",
        ),
        _scene(
            "ending_b_02", "ending_b", "충격적인 대답", "조용한 복도",
            "도현", "너를 버린 게 아니야. 그날, 너를 지키려면 사라져야 했어.",
            "숨겨진 비밀을 암시하는 상대역, 다음 화 예고",
            "He exhales, looks down with regret, then quietly reveals the truth while meeting her eyes. She reacts in shock. Intimate cinematic close-up.",
            10, "#f59e0b",
        ),
    ]

    return {
        "title": title,
        "logline": prompt,
        "genre": genre,
        "mood": mood,
        "characters": [
            {"name": "서윤", "role": "주인공", "voice": "Microsoft Heami Desktop"},
            {"name": "도현", "role": "비밀을 가진 상대역", "voice": "Microsoft Heami Desktop"},
            {"name": "민지", "role": "진실을 목격하는 동료", "voice": "Microsoft Heami Desktop"},
        ],
        "common_scenes": common,
        "ending_a": ending_a,
        "ending_b": ending_b,
        "duration": {"common": 40, "ending": 20, "total": 60},
    }
