from __future__ import annotations

import argparse

import main as main_module


def test_run_is_scene_plot3d_compatibility_alias(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(main_module, "cmd_stream", lambda args: captured.update(vars(args)))
    main_module.cmd_run(
        argparse.Namespace(
            config="scene.json",
            joint_unit="deg",
            joint_mode="dexalign",
            pose_filter_enabled=False,
            rate_hz=30.0,
            duration_sec=2.0,
            max_samples=10,
            plot_fps=15.0,
            plot_range_m=0.6,
            no_skeleton=True,
        )
    )
    assert captured["config"] == "scene.json"
    assert captured["show_plot3d"] is True
    assert captured["show_overlay"] is False
    assert captured["no_stdout"] is True
    assert captured["pose_filter_enabled"] is False
    assert captured["plot_fps"] == 15.0
    assert captured["no_skeleton"] is True


def test_stream_pose_filter_cli_overrides(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(main_module, "cmd_stream", lambda args: captured.update(vars(args)))

    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "stream", "--no-pose-filter", "--max-samples", "1"],
    )
    main_module.main()
    assert captured["pose_filter_enabled"] is False

    monkeypatch.setattr(
        "sys.argv",
        ["main.py", "stream", "--pose-filter", "--max-samples", "1"],
    )
    main_module.main()
    assert captured["pose_filter_enabled"] is True
