from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "umi_mono"
    / "scripts_slam_pipeline_ros2"
    / "00_process_videos_ros2.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("process_videos_ros2", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_bytes(path: Path, size: int) -> None:
    path.write_bytes(b"0" * size)


def test_move_largest_video_to_mapping_ignores_aurco(tmp_path: Path) -> None:
    module = _load_module()
    session_dir = tmp_path / "session"
    demos_dir = session_dir / "demos"
    demos_dir.mkdir(parents=True)

    aurco_demo = demos_dir / "aurco"
    data_001_demo = demos_dir / "data_001"
    data_002_demo = demos_dir / "data_002"
    for path in (aurco_demo, data_001_demo, data_002_demo):
        path.mkdir()

    _write_bytes(aurco_demo / "raw_video.mp4", 900)
    _write_bytes(data_001_demo / "raw_video.mp4", 100)
    _write_bytes(data_002_demo / "raw_video.mp4", 500)
    (aurco_demo / "imu_data.json").write_text("{}", encoding="utf-8")
    (data_001_demo / "imu_data.json").write_text("{}", encoding="utf-8")
    (data_002_demo / "imu_data.json").write_text("{}", encoding="utf-8")

    (session_dir / "aurco").mkdir()
    (session_dir / "data_001").mkdir()
    (session_dir / "data_002").mkdir()

    module.move_largest_video_to_mapping(demos_dir=demos_dir, session_dir=session_dir)

    mapping_dir = demos_dir / "mapping"
    assert (mapping_dir / "raw_video.mp4").is_file()
    assert (mapping_dir / "imu_data.json").is_file()

    assert not data_002_demo.exists()
    assert data_001_demo.exists()
    assert aurco_demo.exists()

    assert not (session_dir / "data_002").exists()
    assert (session_dir / "mapping").is_dir()
    assert (session_dir / "aurco").is_dir()


def test_build_process_args_keeps_explicit_mapping_bag(tmp_path: Path) -> None:
    module = _load_module()
    session_dir = tmp_path / "session"
    demo_dir = session_dir / "demos"
    demo_dir.mkdir(parents=True)

    bag_dirs = []
    for name in ("mapping", "data_001", "aurco"):
        path = session_dir / name
        path.mkdir()
        bag_dirs.append(path)

    process_args = module.build_process_args(
        bag_dirs=bag_dirs,
        demo_dir=demo_dir,
        image_topic="/img",
        accel_topic="/accel",
        gyro_topic="/gyro",
        imu_topic="/imu",
    )
    names = [args[0].name for args in process_args]
    assert names == ["mapping", "data_001", "aurco"]
