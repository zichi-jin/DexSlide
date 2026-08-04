# DexSlide realtime scene API

`DexSlideScene` 统一读取共享相机、table ArUco 和每只手独立的串口、marker body 与 DexAlign 标定。采集核心不创建窗口，也不会自动保存数据。

```python
from dexslide.streaming import DexSlideScene

with DexSlideScene.from_file("assets/dexslide_streaming.json") as scene:
    for sample in scene.samples(rate_hz=30):
        left = sample.hands["left"]
        controller.update(
            transform_world_hand=left.transform_table_hand,
            joint_angles=left.joint_angles,
            valid=left.valid,
        )
```

默认关节单位是 `deg`，位移单位固定为 `m`。`joint_angles_raw` 是 glove calibration 后的值；`joint_angles_dexalign` 是应用 `scale * raw + bias` 后的值；`joint_angles` 由 `joint_mode` 选择。

命令行默认不向 stdout 连续输出 sample；需要 shell JSONL 时显式传入 `--stdout`：

```bash
python main.py stream --config assets/dexslide_streaming.json
python main.py stream --stdout
python main.py stream --joint-mode raw --joint-unit rad
python main.py stream --no-pose-filter
python main.py stream --show-overlay
python main.py stream --show-plot3d
python main.py stream --show-overlay --show-plot3d
python main.py stream --save-dir assets/datasets --session-id demo_001
python main.py stream --show-overlay --show-plot3d --save-dir assets/datasets
```

位姿滤波默认由配置中的 `stream.pose_filter_enabled` 控制。CLI 的
`--pose-filter` / `--no-pose-filter` 会覆盖配置；Python API 可传入
`DexSlideScene.from_file(..., pose_filter_enabled=False)`。关闭后会同时绕过 table
时序平滑、marker-body One Euro 和腕部后置 SE(3) 平滑，但仍保留 ArUco
正反面、朝向和右手系候选筛选。

完整能力边界、CLI 参数和其他脚本索引见 [DexSlide User Guide](USER_GUIDE.md)。

录制结果使用分块 NPZ：

```text
session_id/
├── session_meta.json
├── first_valid_frame.jpg
├── configs/
└── chunks/
    ├── 000000.npz
    └── 000001.npz
```

`session_meta.json` 保存所有实际配置副本的原路径、session 路径和 SHA-256，并记录最终采用的 marker-to-hand transform、DexAlign scale/bias、joint order 和单位。可使用 `DexSlideDatasetReader` 回读：

```python
from dexslide.recording import DexSlideDatasetReader

for sample in DexSlideDatasetReader("assets/datasets/demo_001").iter_samples():
    print(sample.timestamp, sample.hands["left"].joint_angles)
```
