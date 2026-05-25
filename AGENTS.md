# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the CLI entrypoint for calibration, raw serial inspection, and live reconstruction. Core Python code lives in `dexslide/`: `calibration/` for offline A4 + skeleton processing, `kinematics/` for hand math, `visualization/` for Matplotlib viewers, `vision/` for ArUco tracking, and `world_pose/` for ROS/SLAM pose subscription. Runtime data and sample assets live under `assets/` (`photos/`, `skeletons/`, `calibration/`, `mano/`). Use `scripts/` for standalone helpers such as `glove_live_mano.py` and URDF retargeting. `tests/` currently covers `world_pose`; `umi_mono/` is a separate headset-tracking workspace with its own guide.

## Build, Test, and Development Commands
Create an isolated environment before working:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Common entrypoints:
- `python main.py calibrate-skeleton --input-dir assets/photos` extracts a hand skeleton from A4 reference photos.
- `python main.py run --port /dev/ttyACM0` starts live 3D reconstruction from the glove stream.
- `python main.py raw --port /dev/ttyACM0` prints raw 20-channel serial values for debugging.
- `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_slam_pose_subscriber.py -v` runs the current unit suite without host-level pytest plugins.

## Coding Style & Naming Conventions
Use 4-space indentation and keep Python style close to the existing codebase: snake_case for modules, functions, variables, and CLI flags; PascalCase for classes. Prefer `pathlib.Path`, small focused helpers, and explicit error messages in CLI paths. The root repo does not define a formatter or linter, so keep imports, docstrings, and typing consistent with nearby files instead of reformatting broadly.

## Testing Guidelines
Add new tests under `tests/` as `test_<feature>.py`. Prefer deterministic math, parsing, and transformation tests that do not require hardware. For serial, camera, or ROS-facing code, isolate device I/O behind helpers and mock inputs in tests. `tests/test_slam_pose_subscriber.py` depends on ROS 2 Humble packages such as `rclpy` and is most reliable in a ROS-compatible Python 3.10 environment.

## Commit & Pull Request Guidelines
Recent history mixes short imperative messages with optional prefixes such as `feat:` and `chore:`. Follow `scope: summary` when practical, for example `vision: add ArUco hand group parsing`. Keep PRs narrow and include: what changed, how you validated it, hardware assumptions, and screenshots or logs for visualization changes. Do not commit local virtualenvs, cached Python files, or machine-specific calibration variants such as `assets/calibration/*.local.json`.
