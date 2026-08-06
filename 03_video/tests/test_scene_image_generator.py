from studio.scene_image_generator import SceneImageGenerator


def _story(logline: str) -> dict:
    return {
        "logline": logline,
        "visual_bible": "Two astronauts in spacesuits inside a spacecraft.",
        "hero_visual_prompt": "Two astronauts beside a rocket at dusk.",
        "genre": "romance",
        "mood": "tense",
    }


def test_user_premise_and_hero_visual_are_present() -> None:
    story = _story("우주비행사들의 연애")

    prompt = SceneImageGenerator._prompt(story)

    assert story["logline"] in prompt
    assert story["hero_visual_prompt"] in prompt
    assert "every essential subject" in prompt


def test_office_is_not_globally_forbidden() -> None:
    story = _story("서울 사무실에서 벌어지는 미스터리")
    story["hero_visual_prompt"] = "A detective searches a dark office."

    prompt = SceneImageGenerator._prompt(story)

    assert "office" in prompt


def test_story_generates_only_one_image(tmp_path) -> None:
    story = _story("우주비행사들의 연애")
    scene_groups = {
        "common_scenes": 3,
        "ending_a": 1,
        "ending_b": 1,
    }
    for group, count in scene_groups.items():
        story[group] = [
            {"visual_prompt": f"Unique {group} scene {index}"}
            for index in range(count)
        ]

    generator = object.__new__(SceneImageGenerator)

    class FakeImage:
        def convert(self, _mode):
            return self

        def save(self, path, format):
            path.write_bytes(format.encode("ascii"))

    class FakePipeline:
        def __init__(self):
            self.prompts = []

        def __call__(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            return type("Result", (), {"images": [FakeImage()]})()

    pipeline = FakePipeline()
    generator._load_pipeline = lambda: pipeline
    outputs = generator.generate(story, tmp_path / "output")

    assert len(outputs) == 1
    assert len(pipeline.prompts) == 1
    assert story["hero_visual_prompt"] in pipeline.prompts[0]
    assert outputs[0].name == "story_01.png"


def test_image_is_generated_fresh_on_every_request(tmp_path) -> None:
    story = _story("우주비행사들의 연애")
    story["common_scenes"] = [{"visual_prompt": "Two astronauts meet."}]
    story["ending_a"] = [{"visual_prompt": "Ending A"}]
    story["ending_b"] = [{"visual_prompt": "Ending B"}]

    class FakeImage:
        def convert(self, _mode):
            return self

        def save(self, path, format):
            path.write_bytes(format.encode("ascii"))

    class FakePipeline:
        def __init__(self):
            self.calls = 0

        def __call__(self, _prompt, **_kwargs):
            self.calls += 1
            return type("Result", (), {"images": [FakeImage()]})()

    generator = object.__new__(SceneImageGenerator)
    pipeline = FakePipeline()
    generator._load_pipeline = lambda: pipeline
    generator.generate(story, tmp_path / "first")
    generator.generate(story, tmp_path / "second")

    assert pipeline.calls == 2
