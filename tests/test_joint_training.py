"""Joint-stage configuration and actual launcher preflight; no model is loaded."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# common.py uses fcntl only when launching work. These tests exercise --dry-run.
if sys.platform == "win32":
    sys.modules.setdefault("fcntl", types.ModuleType("fcntl"))
from joint_training import joint_config
import train as launcher


class JointTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="recurft-joint-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / "checkpoint-59600"
        self.ckpt.mkdir()
        self.metadata = dict(loop_start_layer=29, loop_end_layer=30, t_lora_rank=128,
                             boundary_head_rank=256, token_conditioning_rank=0,
                             last_layer=35, projection_layer_ids=list(range(29, 36)))
        self.save_meta()
        (self.ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 59600}))
        for name in ("adapter_config.json", "adapter_model.safetensors", "recurft_recurrent.safetensors",
                     "tokenizer_config.json", "tokenizer.json"):
            (self.ckpt / name).write_text("{}")
        self.model_config = {"model_type": "qwen3", "num_hidden_layers": 36}
        self.model = self.root / "base"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps(self.model_config))
        self.data = self.root / "math.json"
        self.data.write_text('[{"query":"q","response":"a"}]')
        self.source = dict(use_recurft=True, finetuning_type="lora", template="qwen3_nothink",
                           recurft_loop_start_layer=29, recurft_loop_end_layer=30,
                           recurft_t_lora_rank=128, recurft_boundary_head_rank=256,
                           recurft_stage1_heads_only=True, recurft_pre_lora_rank=8,
                           recurft_last_lora_rank=8, output_dir="old-output")
        self.recipe = yaml.safe_load((ROOT / "configs/experimental/recurft_joint_boundary.yaml").read_text())
        self.source_path = self.root / "train.yaml"
        self.source_path.write_text(yaml.safe_dump(self.source))
        self.output = self.root / "new-output"

    def save_meta(self):
        (self.ckpt / "recurft_config.json").write_text(json.dumps(self.metadata))

    def build(self, steps=2000):
        return joint_config(self.source, self.recipe, self.ckpt, steps, self.model_config)

    def run_cli(self, extra=(), stage="joint"):
        args = ["train.py", "--stage", stage, "--model", str(self.model), "--data", str(self.data),
                "--output", str(self.output), "--checkpoint", str(self.ckpt), "--gpu", "0",
                "--steps", "2000", "--dry-run"]
        if stage == "joint":
            args += ["--source-config", str(self.source_path)]
        out = io.StringIO()
        with patch.object(sys, "argv", args + list(extra)), redirect_stdout(out), redirect_stderr(io.StringIO()):
            launcher.main()
        return yaml.safe_load("\n".join(out.getvalue().splitlines()[:-1]))

    def test_joint_opens_t_and_head_while_freezing_target(self):
        config = self.build()
        self.assertTrue(config["recurft_recurrent_trainable_only"])
        self.assertFalse(config["recurft_stage1_heads_only"])
        self.assertFalse(config["recurft_multistep_detach_rollout"])
        self.assertEqual(config["recurft_multistep_steps"], 2)
        self.assertGreater(config["recurft_recurrent_loss_weight"], 0)
        self.assertAlmostEqual(config["recurft_multistep_loss_weight"] *
                               config["recurft_multistep_boundary_logit_kl_loss_weight"], 1.0)
        self.assertTrue(self.source["recurft_stage1_heads_only"])

    def test_continuation_steps_and_warmup_start(self):
        (self.ckpt / "trainer_state.json").write_text(json.dumps({"global_step": 33025}))
        config = self.build(1200)
        self.assertEqual(config["max_steps"], 34225)
        self.assertEqual(config["recurft_multistep_warmup_start_step"], 33025)

    def test_trainable_target_source_does_not_leak_into_frozen_continuation(self):
        self.source['recurft_joint_mode'] = 'trainable_target'
        self.assertEqual(self.build()['recurft_joint_mode'], 'legacy')
        self.assertTrue(self.build()['recurft_recurrent_trainable_only'])

    def test_safe_optimizer_cannot_cross_into_a_new_scope(self):
        path = self.ckpt/'optimizer_safe.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'optimizer groups'):
            self.build()
        self.assertEqual(path.read_text(), '{}')

    def test_requires_boundary_checkpoint(self):
        self.metadata["boundary_head_rank"] = 0
        self.save_meta()
        with self.assertRaisesRegex(ValueError, "boundary-warmup"):
            self.build()

    def test_mismatched_source_layout_is_rejected(self):
        self.source["recurft_loop_start_layer"] = 33
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.build()

    def test_wrong_model_depth_is_rejected(self):
        self.model_config["num_hidden_layers"] = 32
        with self.assertRaisesRegex(ValueError, "layer count"):
            self.build()

    def test_old_optimizer_is_not_loaded_or_deleted(self):
        path = self.ckpt / "optimizer.pt"
        path.write_bytes(b"keep-original")
        with self.assertRaisesRegex(ValueError, "optimizer groups"):
            self.build()
        self.assertEqual(path.read_bytes(), b"keep-original")

    def test_invalid_step_counts(self):
        for value in (None, 0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(value)

    def test_dry_run_materializes_correct_config_without_writes(self):
        before = {p.name: p.read_bytes() for p in self.ckpt.iterdir()}
        config = self.run_cli()
        self.assertEqual(config["max_steps"], 61600)
        self.assertEqual(config["resume_from_checkpoint"], str(self.ckpt.resolve()))
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertFalse(config["overwrite_output_dir"])
        self.assertEqual(config["output_dir"], str(self.output.resolve() / "checkpoint_output"))
        self.assertFalse(self.output.exists())
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.ckpt.iterdir()})

    def test_qwen_llama_template_is_rejected(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli(["--template", "llama3"])
        self.assertEqual(caught.exception.code, 2)

    def test_existing_output_is_rejected(self):
        self.output.mkdir()
        with self.assertRaises(SystemExit):
            self.run_cli()

    def test_existing_warmup_accepts_explicit_qwen_template(self):
        config = self.run_cli(["--template", "qwen3_nothink"], stage="boundary-warmup")
        self.assertEqual(config["template"], "qwen3_nothink")
        self.assertEqual(config["max_steps"], 61600)
        self.assertTrue(config["recurft_stage1_heads_only"])


if __name__ == "__main__":
    unittest.main()
