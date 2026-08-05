from studio.scene_image_generator import SceneImageGenerator


def _story(logline: str) -> dict:
    return {
        "logline": logline,
        "visual_bible": "Two astronauts in spacesuits inside a spacecraft.",
        "genre": "romance",
        "mood": "tense",
    }


def test_user_premise_and_exact_scene_are_present() -> None:
    story = _story("우주비행사들의 연애")
    scene = {"visual_prompt": "Two astronauts beside a rocket at dusk."}

    prompt = SceneImageGenerator._prompt(story, scene, 1)

    assert story["logline"] in prompt
    assert scene["visual_prompt"] in prompt
    assert "location must match" in prompt


def test_office_is_not_globally_forbidden() -> None:
    story = _story("서울 사무실에서 벌어지는 미스터리")
    scene = {"visual_prompt": "A detective searches a dark office."}

    prompt = SceneImageGenerator._prompt(story, scene, 1)
    negative = SceneImageGenerator._negative_prompt()

    assert "office" in prompt
    assert "corporate interior" not in negative


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

    class FakeClient:
        def __init__(self):
            self.prompts = []

        def text_to_image(self, prompt, **_kwargs):
            self.prompts.append(prompt)
            return FakeImage()

    generator.client = FakeClient()
    outputs = generator.generate(story, tmp_path / "output")

    assert len(outputs) == 1
    assert len(generator.client.prompts) == 1
    assert "Shot 1:" in generator.client.prompts[0]
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

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def text_to_image(self, _prompt, **_kwargs):
            self.calls += 1
            return FakeImage()

    generator = object.__new__(SceneImageGenerator)
    generator.client = FakeClient()
    generator.generate(story, tmp_path / "first")
    generator.generate(story, tmp_path / "second")

    assert generator.client.calls == 2
