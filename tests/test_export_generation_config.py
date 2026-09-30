"""CPU regressions for native OLMo generation defaults and the Qwen policy."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from transformers import GenerationConfig, Olmo2Config, Qwen2Config, Qwen3Config

# The package __init__ eagerly imports CUDA training kernels. Load the real
# exporter module directly: generation metadata itself needs no CUDA driver.
_spec = importlib.util.spec_from_file_location(
    "loopify_export_under_test", Path(__file__).resolve().parents[1] / "src/loopify/export_hf.py"
)
_exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exporter)
export_generation_config = _exporter.export_generation_config


class Tokenizer:
    unk_token_id = 100257

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 100265


class ExportGenerationConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)

    def test_olmo_retains_generation_config_even_with_chatml_tokens(self):
        reference = GenerationConfig(
            eos_token_id=100257, pad_token_id=100277, do_sample=False,
            max_new_tokens=321, repetition_penalty=1.15,
        )
        reference.save_pretrained(self.path)
        config = Olmo2Config(eos_token_id=100257, pad_token_id=100277)
        result = export_generation_config(str(self.path), config, Tokenizer())
        for key in ("eos_token_id", "pad_token_id", "do_sample", "max_new_tokens", "repetition_penalty"):
            self.assertEqual(getattr(result, key), getattr(reference, key))
        self.assertNotEqual(result.eos_token_id, [100265, 100257])
        # Check the serialized configuration written alongside the exported model.
        destination = self.path / "export"
        result.save_pretrained(destination)
        reloaded = GenerationConfig.from_pretrained(destination)
        self.assertEqual(reloaded.eos_token_id, 100257)
        self.assertEqual(reloaded.pad_token_id, 100277)
        self.assertFalse(reloaded.do_sample)

    def test_olmo_missing_generation_config_uses_model_defaults(self):
        config = Olmo2Config(eos_token_id=100257, pad_token_id=100277)
        result = export_generation_config(str(self.path), config, Tokenizer())
        self.assertEqual(result.eos_token_id, 100257)
        self.assertEqual(result.pad_token_id, 100277)
        self.assertFalse(result.do_sample)

    def test_invalid_existing_generation_config_is_not_hidden(self):
        (self.path / "generation_config.json").write_text("not valid JSON")
        with self.assertRaises(OSError):
            export_generation_config(str(self.path), Olmo2Config(), Tokenizer())

    def check_qwen_policy(self, config_class):
        # Existing Qwen policy deliberately overrides differing reference defaults.
        GenerationConfig(eos_token_id=42, pad_token_id=43, do_sample=False).save_pretrained(self.path)
        config = config_class(bos_token_id=151643, eos_token_id=151643)
        tokenizer = SimpleNamespace(unk_token_id=None, convert_tokens_to_ids=lambda _: 151645)
        result = export_generation_config(str(self.path), config, tokenizer)
        self.assertEqual(result.bos_token_id, 151643)
        self.assertEqual(result.eos_token_id, [151645, 151643])
        self.assertTrue(result.do_sample)
        self.assertEqual(result.temperature, 0.6)
        self.assertEqual(result.top_p, 0.95)
        self.assertIsNone(result.pad_token_id)

    def test_qwen2_keeps_turn_end_and_sampling_policy(self):
        self.check_qwen_policy(Qwen2Config)

    def test_qwen3_keeps_turn_end_and_sampling_policy(self):
        self.check_qwen_policy(Qwen3Config)

    def check_qwen_stops(self, im_end):
        config = Qwen3Config(bos_token_id=151643, eos_token_id=[151645, 151643])
        tokenizer = SimpleNamespace(unk_token_id=0, convert_tokens_to_ids=lambda _: im_end)
        result = export_generation_config(str(self.path), config, tokenizer)
        self.assertEqual(result.eos_token_id, [151645, 151643])

    def test_qwen_does_not_duplicate_existing_turn_end(self):
        self.check_qwen_stops(151645)

    def test_qwen_does_not_add_unknown_stop_token(self):
        self.check_qwen_stops(0)


if __name__ == "__main__":
    unittest.main()
