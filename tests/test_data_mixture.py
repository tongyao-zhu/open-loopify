"""Offline mixture integration tests: tiny token files, no GPU or downloads."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("prepare_tokens", "data/prepare_tokens.py")
config = load("make_config", "configs/make_config.py")


def args(**kwargs):
    values = dict(mixture="reasoning-v1", sources=None, shards=None, max_doc_tokens=None,
                  model="test/model", out=None)
    return SimpleNamespace(**(values | kwargs))


class Tokenizer:
    eos_token_id = 0

    def __call__(self, texts, **kwargs):
        return SimpleNamespace(input_ids=[[1] * 8 for _ in texts])


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "data"
        self.out.mkdir()

    def prepare_tiny(self):
        a = args(out=str(self.out))
        jobs, recipe = prepare.preparation_plan(a)
        jobs = [job | {"budget": 24} for job in jobs]
        fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: Tokenizer()))
        with patch.dict(sys.modules, {"transformers": fake}), patch.object(
            prepare, "docs", side_effect=lambda *a: iter(["example " * 8] * 2)
        ):
            for job in jobs:
                prepare.run(job["source"], job["budget"], a.out, a.model, job["max_doc_tokens"], job["shard"])
        manifest = prepare.write_manifest(jobs, a, recipe)
        return jobs, a, recipe, manifest

    def test_published_recipe_preserves_budgets_filters_and_source_ratio(self):
        jobs, _ = prepare.preparation_plan(args())
        self.assertEqual(len(jobs), 9)
        self.assertEqual(sum(j["budget"] for j in jobs[:8]), 3_200_000_000)
        self.assertEqual(jobs[-1]["budget"], 600_000_000)
        self.assertTrue(all(j["max_doc_tokens"] == 15000 for j in jobs[:8]))
        self.assertEqual(jobs[-1]["max_doc_tokens"], 0)
        weights = [j["sampling_weight"] for j in jobs]
        historical = [175] * 8 + [700]
        self.assertEqual([w / sum(weights) for w in weights], [w / sum(historical) for w in historical])

    def test_custom_source_cli_remains_supported(self):
        jobs, recipe = prepare.preparation_plan(args(mixture=None, sources="openr1math=600", shards=2))
        self.assertIsNone(recipe)
        self.assertEqual([j["budget"] for j in jobs], [300_000_000] * 2)
        self.assertEqual([j["shard"] for j in jobs], [(0, 2), (1, 2)])
        self.assertEqual(jobs[0]["max_doc_tokens"], 15000)

    def test_ambiguous_recipe_overrides_are_rejected(self):
        for overrides in [dict(sources="openr1math=1"), dict(shards=2), dict(max_doc_tokens=0)]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                prepare.preparation_plan(args(**overrides))

    def test_empty_budget_and_unknown_recipe_are_rejected(self):
        for a in [args(mixture="missing"), args(mixture=None, sources="openr1math=0")]:
            with self.assertRaises(ValueError):
                prepare.preparation_plan(a)

    def test_preprocessing_to_config_roundtrip_and_relocation(self):
        jobs, a, recipe, manifest = self.prepare_tiny()
        self.assertEqual(manifest["total_tokens"], 9 * 18)
        relocated = self.out.with_name("relocated")
        shutil.move(self.out, relocated)
        folders, weights = config.parse_data([str(relocated)], expected_tokenizer=a.model)
        self.assertEqual([Path(f).name for f in folders], [prepare.job_name(j) for j in jobs])
        self.assertTrue(all(Path(f).parent == relocated for f in folders))
        self.assertEqual(weights, [0.25] * 8 + [1.0])
        training_args = SimpleNamespace(loop="13:22:3", gpus=8, tp=1, train_steps=3000,
            name="test", tag="", seed=17, run_root="runs", ckpt="base", ckpt_every=500,
            zero1=True, recompute=True, seq_len=16384, lr=3e-5, hf=a.model)
        loop = config.build_config(training_args, "loop", {"vocab_size": 128}, folders, weights)
        dense = config.build_config(training_args, "dense", {"vocab_size": 128}, folders, weights)
        dataset = loop["data_stages"][0]["data"]["dataset"]
        self.assertEqual(dataset["dataset_weights"], weights)
        self.assertEqual(dataset["dataset_folder"], folders)
        self.assertEqual(loop["data_stages"], dense["data_stages"])

    def test_mismatched_tokenizer_is_rejected(self):
        jobs, a, _, _ = self.prepare_tiny()
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            config.parse_data([str(self.out)], expected_tokenizer="another/model")
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            prepare.check_prepared_job(jobs[0], a.out, "another/model", allow_missing=True)

    def test_matching_prepared_shard_is_reusable_but_changed_budget_is_not(self):
        jobs, a, _, _ = self.prepare_tiny()
        prepare.check_prepared_job(jobs[0], a.out, a.model, allow_missing=True)
        with self.assertRaises(ValueError):
            prepare.check_prepared_job(jobs[0] | {"budget": 100}, a.out, a.model, allow_missing=True)

    def test_incomplete_data_cannot_be_published_or_consumed(self):
        jobs, a, recipe, _ = self.prepare_tiny()
        name = prepare.job_name(jobs[0])
        (self.out / name / f"{name}.ds").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            prepare.write_manifest(jobs, a, recipe)
        with self.assertRaises(ValueError):
            config.parse_data([str(self.out)])

    def test_missing_shard_cannot_publish_manifest(self):
        jobs, recipe = prepare.preparation_plan(args())
        with self.assertRaises(ValueError):
            prepare.write_manifest(jobs, args(out=str(self.out)), recipe)
        self.assertFalse((self.out / "manifest.json").exists())

    def test_manifest_weights_must_match_shards(self):
        _, _, _, manifest = self.prepare_tiny()
        del manifest["sampling_weights"]["openr1math"]
        (self.out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            config.parse_data([str(self.out)])

    def test_mixture_weight_override_is_rejected(self):
        self.prepare_tiny()
        with self.assertRaisesRegex(ValueError, "supplies its own weights"):
            config.parse_data([f"{self.out}:700"])

    def test_individual_folder_weights_remain_supported(self):
        self.prepare_tiny()
        folders, weights = config.parse_data([str(self.out / "openthoughts3_p*") + ":175",
                                             str(self.out / "openr1math") + ":700"])
        self.assertEqual(weights, [175] * 8 + [700])
        self.assertEqual(len(folders), 9)
        _, weights = config.parse_data([str(self.out / "openr1math")])
        self.assertEqual(weights, [18])


if __name__ == "__main__":
    unittest.main()
