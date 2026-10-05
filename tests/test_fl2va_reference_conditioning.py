"""CPU-only tests for FL2VA image references alongside temporal keyframes."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_conditioning_function():
    tree = ast.parse((ROOT / "fl2va_engine.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_fl2va_conditioning"
    )

    class Helpers:
        @staticmethod
        def conditioning_set_values(conditioning, values):
            for _embedding, metadata in conditioning:
                metadata.update(values)
            return conditioning

    namespace = {
        "_align_frame_count": lambda count: int(count),
        "_empty_av_latent": lambda width, height, count: {"width": width, "height": height, "frames": count},
        "_resize": lambda image, width, height, crop: (image, width, height, crop),
        "node_helpers": Helpers,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fl2va_engine.py", "exec"), namespace)
    return namespace["make_fl2va_conditioning"]


class FakeClip:
    def tokenize(self, prompt, **kwargs):
        self.prompt = prompt
        self.tokenize_kwargs = kwargs
        return {"prompt": prompt, **kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        self.tokens = tokens
        return [["conditioning", {}]]


class FakeVae:
    def encode(self, image):
        return ("vae", image)


class Fl2vaReferenceConditioning(unittest.TestCase):
    def test_identity_reference_and_first_last_temporal_frames_coexist(self):
        build = load_conditioning_function()
        clip = FakeClip()
        reference = "character reference"
        first = ["first frame"]
        last = ["last frame"]

        conditioning, latent = build(
            clip,
            FakeVae(),
            "A person matching <Picture 1> walks through the scene.",
            288,
            512,
            97,
            first_frame=first,
            last_frame=last,
            reference_images=[reference],
        )

        images = clip.tokenize_kwargs["images"]
        self.assertEqual(images[0], reference)
        self.assertEqual([entry[0] for entry in images[1:]], [first, last])
        keyframes = conditioning[0][1]["minimax_keyframes"]
        self.assertEqual([item["resolved_frame_index"] for item in keyframes], [0, 96])
        self.assertEqual(keyframes[0]["latent"][1][3], "disabled")
        self.assertEqual(keyframes[1]["latent"][1][3], "center")
        self.assertEqual(latent, {"width": 288, "height": 512, "frames": 97})

    def test_reference_only_does_not_create_temporal_keyframes(self):
        build = load_conditioning_function()
        clip = FakeClip()

        conditioning, _latent = build(
            clip,
            FakeVae(),
            "Use the character reference.",
            288,
            512,
            97,
            reference_images=["character reference"],
        )

        self.assertEqual(clip.tokenize_kwargs["images"], ["character reference"])
        self.assertNotIn("minimax_keyframes", conditioning[0][1])


if __name__ == "__main__":
    unittest.main()
