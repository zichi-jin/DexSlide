# Repository Guidelines

## Project Structure & Module Organization
- `umi/`: core robotics utilities (pose math, real-world device control, shared memory, trajectory eval).
- `diffusion_policy/`: learning stack (Hydra configs, datasets, models, workspaces, env wrappers).
- `scripts/`, `scripts_real/`, `scripts_slam_pipeline_ours/`: operational scripts for calibration, deployment, and SLAM/dataset pipelines.
- `example/`: sample robot and calibration configs; `config/`: camera calibration YAMLs.
- `tests/`: script-style test files (`test_*.py`), plus top-level entrypoints like `train.py`, `run_slam_pipeline.py`, and `eval_real.py`.

## Build, Test, and Development Commands
- Create environment: `mamba env create -f conda_environment.yaml`
- Activate: `conda activate umi`
- Run SLAM pipeline: `python run_slam_pipeline.py <session_dir>`
- Train (single GPU):  
  `python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=<dataset.zarr.zip>`
- Train (multi GPU):  
  `accelerate --num_processes <ngpus> train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=<dataset.zarr.zip>`
- Quick functional test: `python tests/test_pose_util.py`

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and snake_case for modules/functions/variables; classes use PascalCase.
- Keep scripts task-oriented and explicit. For pipeline stages, follow existing numbered pattern (for example `06_generate_dataset_plan.py`).
- Prefer small, focused functions and avoid hidden side effects in utility modules.
- `black` is pinned in `requirements.txt`; format touched Python files with `python -m black <paths>`.

## Testing Guidelines
- There is no enforced CI test matrix in this repo; validate changes with targeted scripts.
- Add new tests under `tests/` using `test_<feature>.py`.
- Hardware-dependent tests (for example UVC/multi-camera tests) require connected devices and may write into `data_local/`.
- For non-hardware logic, provide deterministic assertions similar to `tests/test_pose_util.py`.

## Commit & Pull Request Guidelines
- Existing history uses short, imperative subjects (`add ...`, `update ...`, `fix ...`). Keep commit titles concise and action-first.
- Recommended pattern: `<area>: <what changed>` (example: `slam: fix UVC camera index assignment`).
- PRs should include: scope, commands run, hardware/setup assumptions, and before/after evidence for behavior changes (logs, screenshots, or short GIFs).
- Do not commit generated artifacts or local run data (`data/`, `data_local/`, `outputs/`, `wandb/`, `example_demo_session/`).
