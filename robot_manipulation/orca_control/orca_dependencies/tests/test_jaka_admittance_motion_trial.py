import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "jaka_admittance_motion_trial.py"

spec = importlib.util.spec_from_file_location("jaka_admittance_motion_trial", MODULE_PATH)
trial_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = trial_module
spec.loader.exec_module(trial_module)


def test_choose_payload_identify_target_changes_only_wrist_joints():
    start = [0.0, 0.1, -0.2, 0.3, -0.4, 0.5]
    target = trial_module.choose_payload_identify_target(start, 30.0)
    assert target[:3] == start[:3]
    assert target[3] != start[3]
    assert target[4] != start[4]
    assert target[5] != start[5]


def test_parse_identified_payload_accepts_flat_format():
    payload = trial_module.parse_identified_payload((0, (0, 1.5, 10.0, 20.0, 30.0)))
    assert payload == trial_module.IdentifiedPayload(1.5, (10.0, 20.0, 30.0))


def test_parse_identified_payload_accepts_nested_centroid_format():
    payload = trial_module.parse_identified_payload((0, (99, 2.5, [1.0, 2.0, 3.0])))
    assert payload == trial_module.IdentifiedPayload(2.5, (1.0, 2.0, 3.0))


def test_load_saved_payload_snapshot_returns_none_for_invalid_file(tmp_path, monkeypatch):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"valid": false}\n', encoding="utf-8")
    monkeypatch.setattr(trial_module, "PAYLOAD_CONFIG_PATH", payload_path)
    assert trial_module.load_saved_payload_snapshot() is None


def test_load_saved_payload_snapshot_reads_valid_file(tmp_path, monkeypatch):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        '{"valid": true, "payload": {"mass_kg": 1.2, "centroid_mm": [4, 5, 6]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(trial_module, "PAYLOAD_CONFIG_PATH", payload_path)
    assert trial_module.load_saved_payload_snapshot() == trial_module.IdentifiedPayload(1.2, (4.0, 5.0, 6.0))


def test_read_force_reading_returns_none_for_bad_shape():
    class Robot:
        def get_torque_sensor_data(self, _source_type):
            return (0, (0, 0, [1.0, 2.0]))

    assert trial_module.read_force_reading(Robot()) is None


def test_read_force_reading_parses_force_tuple():
    class Robot:
        def get_torque_sensor_data(self, _source_type):
            return (0, (0, 0, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))

    assert trial_module.read_force_reading(Robot()) == trial_module.ForceReading(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)


def test_trajectory_offset_stays_within_requested_amplitudes():
    offset = trial_module.trajectory_offset(
        elapsed_s=3.0,
        xy_amplitude_mm=35.0,
        z_amplitude_mm=25.0,
        rotation_amplitude_rad=0.2,
        period_s=24.0,
    )
    assert abs(offset[0]) <= 35.0
    assert abs(offset[1]) <= 35.0
    assert abs(offset[2]) <= 25.0
    assert abs(offset[3]) <= 0.2
    assert abs(offset[4]) <= 0.2
    assert abs(offset[5]) <= 0.2


def test_trajectory_offset_starts_from_zero():
    assert trial_module.trajectory_offset(0.0, 35.0, 25.0, 0.2, 24.0) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
