"""Audio-reference resampling must not be a load-time H3 video dependency."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_audio_encoder(torchaudio):
    tree = ast.parse((ROOT / "extender.py").read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_encode_ref_audio"
    )
    namespace = {"torchaudio": torchaudio}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "extender.py", "exec"), namespace)
    return namespace["_encode_ref_audio"]


class FakeWaveform:
    def __getitem__(self, _key):
        return self

    def movedim(self, _source, _destination):
        return self


class FakeVae:
    audio_sample_rate = 32000

    def encode(self, _waveform):
        return type("Latent", (), {"shape": (1, 32, 2, 7)})()


class OptionalTorchaudio(unittest.TestCase):
    def test_matching_sample_rate_does_not_require_torchaudio(self):
        encode = load_audio_encoder(None)
        latent, length = encode(FakeVae(), {"waveform": FakeWaveform(), "sample_rate": 32000})
        self.assertEqual(latent.shape, (1, 32, 2, 7))
        self.assertEqual(length, 7)

    def test_resampling_without_torchaudio_has_actionable_error(self):
        encode = load_audio_encoder(None)
        with self.assertRaisesRegex(RuntimeError, "torchaudio is required to resample audio-reference input"):
            encode(FakeVae(), {"waveform": FakeWaveform(), "sample_rate": 44100})


if __name__ == "__main__":
    unittest.main()
