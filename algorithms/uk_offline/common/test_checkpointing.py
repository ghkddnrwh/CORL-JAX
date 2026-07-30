import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from algorithms.uk_offline.common.checkpointing import (
    TrainingCheckpointManager,
    load_pickle,
    log_training_exceptions,
    save_logs_npz,
)


class FakeSummary(dict):
    pass


class FakeRun:
    def __init__(self, run_id, entity=None, project=None, summary=None):
        self.id = run_id
        self.entity = entity
        self.project = project
        self.summary = FakeSummary(summary or {})
        self.logs = []
        self.metric_definitions = []

    def log_code(self, root):
        self.code_root = str(root)

    def define_metric(self, *args, **kwargs):
        self.metric_definitions.append((args, kwargs))

    def log(self, metrics):
        payload = dict(metrics)
        self.logs.append(payload)
        # Simulate W&B summary's last-value behavior.
        self.summary.update(payload)


class FakeWandb:
    def __init__(self, existing_summary=None):
        self.inits = []
        self.run = None
        self.existing_summary = dict(existing_summary or {})

    def init(self, **kwargs):
        self.inits.append(kwargs)
        self.run = FakeRun(
            kwargs["id"],
            kwargs.get("entity"),
            kwargs.get("project"),
            summary=self.existing_summary,
        )
        return self.run

    def finish(self, **kwargs):
        self.run = None


class AlwaysFailWandb:
    def __init__(self):
        self.run = None
        self.calls = 0

    def init(self, **kwargs):
        self.calls += 1
        raise RuntimeError(f"wandb unavailable {self.calls}")

    def finish(self, **kwargs):
        self.run = None


class CheckpointingTests(unittest.TestCase):
    def config(self):
        return {
            "env": "dummy-v0",
            "seed": 0,
            "max_timesteps": 10,
            "batch_size": 4,
            "checkpoint_freq": 5,
            "log_wandb": True,
            "project": "test",
            "group": "g",
            "name": "n",
            "checkpoints_path": None,
            "load_model": "",
            "mode": "train",
            "save_final_model": False,
            "wandb_entity": None,
        }

    def manager(self, path, config=None):
        config = self.config() if config is None else config
        return TrainingCheckpointManager(
            run_dir=path,
            current_config=config,
            default_config=self.config(),
            max_timesteps=10,
            checkpoint_type="test_progress",
            checkpoint_version=2,
            accepted_checkpoint_versions=(1, 2),
            wandb_enabled=True,
            wandb_project="test",
        )

    def test_checkpoint_first_then_buffered_wandb_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            self.assertEqual(manager.prepare().mode, "new")
            state = {"step": 0, "weight": np.array([1.0])}

            # The timestep-0 model is committed before W&B exists.
            manager.save_progress(0, state, [])
            checkpoint = load_pickle(Path(tmp) / "latest_checkpoint.pkl")
            self.assertEqual(checkpoint["timestep"], 0)
            self.assertEqual(len(checkpoint["pending_wandb_logs"]), 1)

            wb = FakeWandb()
            manager.initialize_wandb(wb, self.config())
            self.assertEqual(wb.inits[0]["resume"], "allow")
            self.assertIn("id", wb.inits[0])
            self.assertNotIn("resume_from", wb.inits[0])
            self.assertEqual(len(wb.run.logs), 1)
            self.assertEqual(wb.run.logs[0]["training/timestep"], 0)
            self.assertEqual(wb.run.logs[0]["checkpoint/saved"], 1)

            manager.log_wandb({"loss": 2.0}, 2)
            manager.log_wandb({"loss": 1.0, "actor": 3.0}, 4)
            self.assertEqual(len(wb.run.logs), 1, "metrics must remain local before checkpoint")

            state = {"step": 5, "weight": np.array([2.0])}
            manager.save_progress(
                5,
                state,
                [{"timestep": 5, "eval/reward_mean": 1.0}],
            )
            # Two train records plus the checkpoint marker record at step 5.
            self.assertEqual(len(wb.run.logs), 4)
            self.assertEqual(
                [row["training/timestep"] for row in wb.run.logs[-3:]],
                [2, 4, 5],
            )
            self.assertEqual(wb.run.logs[-1]["checkpoint/saved"], 1)

    def test_resume_skips_partially_delivered_wandb_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            manager.save_progress(0, {"step": 0}, [])
            wb = FakeWandb()
            manager.initialize_wandb(wb, self.config())

            manager.log_wandb({"loss": 3.0}, 1)
            manager.log_wandb({"loss": 2.0}, 2)
            manager.log_wandb({"loss": 1.0}, 3)
            # Save while W&B is unavailable, leaving the whole batch in checkpoint.
            manager._wandb_run = None
            manager.save_progress(5, {"step": 5}, [])

            restored = self.manager(tmp)
            self.assertEqual(restored.prepare().mode, "resume")
            holder = {}
            step, _, _ = restored.restore(
                lambda payload: holder.update(payload),
                lambda: holder["step"],
            )
            self.assertEqual(step, 5)
            pending_sequences = [
                item["sequence"]
                for item in load_pickle(Path(tmp) / "latest_checkpoint.pkl")["pending_wandb_logs"]
            ]
            # Pretend W&B accepted the first two rows before the previous process died.
            remote_sequence = pending_sequences[1]
            wb2 = FakeWandb(
                existing_summary={
                    "resume/last_sequence": remote_sequence,
                    "resume/last_checkpoint_timestep": 0,
                }
            )
            restored.initialize_wandb(wb2, self.config())
            sent_sequences = [row["resume/log_sequence"] for row in wb2.run.logs]
            self.assertEqual(
                sent_sequences,
                [seq for seq in pending_sequences if seq > remote_sequence],
            )

    def test_same_run_id_is_reused_with_public_resume_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            manager.save_progress(0, {"step": 0}, [])
            first_id = manager.wandb_state.run_id

            restored = self.manager(tmp)
            restored.prepare()
            holder = {}
            restored.restore(
                lambda payload: holder.update(payload),
                lambda: holder["step"],
            )
            wb = FakeWandb()
            restored.initialize_wandb(wb, self.config())
            self.assertEqual(wb.inits[0]["id"], first_id)
            self.assertEqual(wb.inits[0]["resume"], "allow")
            self.assertNotIn("resume_from", wb.inits[0])

    def test_completed_run_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            manager.save_progress(0, {"step": 0}, [])
            manager.complete(10, {"step": 10}, False, [])
            again = self.manager(tmp)
            self.assertEqual(again.prepare().mode, "completed")

    def test_config_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            manager.save_progress(0, {"step": 0}, [])
            changed = self.config()
            changed["batch_size"] = 8
            with self.assertRaises(ValueError):
                self.manager(tmp, changed).prepare()

    def test_yaml_list_and_python_tuple_configs_are_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config()
            config["actor_hidden_dims"] = (512, 512, 512, 512)
            config["value_hidden_dims"] = (512, 512, 512, 512)

            manager = self.manager(tmp, config)
            self.assertEqual(manager.prepare().mode, "new")
            manager.save_progress(0, {"step": 0}, [])

            # YAML safe_dump/load converts tuples to lists. Reopening with the
            # original tuple-valued Python config must still be considered the
            # same training configuration.
            restored = self.manager(tmp, config)
            self.assertEqual(restored.prepare().mode, "resume")
            holder = {}
            step, _, _ = restored.restore(
                lambda payload: holder.update(payload),
                lambda: holder["step"],
            )
            self.assertEqual(step, 0)

    def test_initialization_metadata_without_checkpoint_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            again = self.manager(tmp)
            self.assertEqual(again.prepare().mode, "new")

    def test_custom_final_saver(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config()
            manager = TrainingCheckpointManager(
                run_dir=tmp,
                current_config=config,
                default_config=self.config(),
                max_timesteps=10,
                checkpoint_type="torch_like",
                checkpoint_version=2,
                accepted_checkpoint_versions=(1, 2),
                wandb_enabled=False,
                final_checkpoint_name="checkpoint.pt",
            )
            manager.prepare()
            manager.save_progress(0, {"step": 0}, [])

            def saver(path, state):
                Path(path).write_bytes(repr(state).encode())

            final_path = manager.complete(10, {"step": 10}, True, [], final_saver=saver)
            self.assertEqual(final_path.name, "checkpoint.pt")
            self.assertTrue(final_path.exists())

    def test_legacy_eval_log_at_max_timesteps_is_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            import yaml

            (tmp_path / "config.yaml").write_text(
                yaml.safe_dump(self.config(), sort_keys=False)
            )
            save_logs_npz(
                [
                    {"timestep": 5, "eval/reward_mean": 1.0},
                    {"timestep": 10, "eval/reward_mean": 2.0},
                ],
                tmp_path / "eval_logs.npz",
            )
            manager = self.manager(tmp)
            preparation = manager.prepare()
            self.assertEqual(preparation.mode, "completed")
            self.assertIn("Legacy completed run", preparation.message)
            status = yaml.safe_load((tmp_path / "training_status.yaml").read_text())
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["timestep"], 10)

    def test_incomplete_legacy_eval_log_is_not_restarted_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            import yaml

            (tmp_path / "config.yaml").write_text(
                yaml.safe_dump(self.config(), sort_keys=False)
            )
            save_logs_npz(
                [{"timestep": 5, "eval/reward_mean": 1.0}],
                tmp_path / "eval_logs.npz",
            )
            with self.assertRaisesRegex(RuntimeError, "Refusing to restart"):
                self.manager(tmp).prepare()

    def test_wandb_double_initialization_failure_disables_only_wandb(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            wb = AlwaysFailWandb()
            output = io.StringIO()
            with redirect_stdout(output):
                initialized = manager.initialize_wandb_with_fallback(
                    wb, self.config()
                )
            self.assertFalse(initialized)
            self.assertFalse(manager.wandb_enabled)
            self.assertEqual(wb.calls, 2)
            self.assertIn("Continuing local training with W&B disabled", output.getvalue())

    def test_completion_warns_when_buffered_wandb_logs_remain(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.manager(tmp)
            manager.prepare()
            manager.log_wandb({"loss": 1.0}, 1)
            output = io.StringIO()
            with redirect_stdout(output):
                manager.complete(10, {"step": 10}, False, [])
            self.assertIn("buffered W&B record(s) were not delivered", output.getvalue())
            self.assertFalse((Path(tmp) / "latest_checkpoint.pkl").exists())

    def test_uncaught_error_is_appended_to_training_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            class Config:
                checkpoints_path = tmp
                load_model = ""

            @log_training_exceptions
            def failing_train(config):
                raise RuntimeError("deliberate failure")

            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                failing_train(Config())
            log_path = Path(tmp) / "training_errors.log"
            self.assertTrue(log_path.exists())
            contents = log_path.read_text()
            self.assertIn("RuntimeError: deliberate failure", contents)
            self.assertIn("traceback:", contents)


if __name__ == "__main__":
    unittest.main()
