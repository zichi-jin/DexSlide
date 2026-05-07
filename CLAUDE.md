# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DexSlide is an open-source exoskeleton data glove for recording dexterous hand manipulation. The core mechanical innovation is the **sliding mechanism** — the glove's linkages stay parallel to finger phalanges without needing to be strapped tight, so it directly measures true joint angles regardless of hand size, and achieves high DOF / range of motion (unlike fixed-structure gloves that require trigonometric correction).

The project has two electrical systems:
- **System A (Joint Sampling)** — current focus. 20 rotary encoders → 5 ADCs → STM32F103RCT6 → USB → PC.
- **System B (Tactile Sampling)** — deferred. Denser sensing, STM32H743XIH6, USB hub (SL6243A).

### Overall Data Pipeline
1. **Skeleton registration** (offline, bare hand): RGB photos + manual A4 corner marking + homography(mm) + MediaPipe landmarks → `software/python/assets/skeletons/skeleton.json`
2. **Zero-pose calibration** (glove on, hand flat on surface): Record baseline angles → `zero_pose.json`
3. **Live use** (glove on): Stream 20 joint angles → forward kinematics with DH params → real-time 3D hand reconstruction
4. **World-frame localization** (planned): Marker fixed to palm dorsum, tracked by external camera → palm pose in world coordinates. No arm IK needed.

### Research Context

- **Retargeting**: Once the hand is reconstructed kinematically, the goal is to map from this glove's configuration (m1) to a robot hand (m2) via **end-effector pose alignment** (not joint-angle remapping or keypoint-distance minimization like BunnyTeleop). This is theoretically more principled than keypoint alignment, though cross-device tactile transfer remains an open problem.
- **Why a glove instead of finger-tip marker tracking?** (1) Severe occlusion during dexterous manipulation; (2) the glove's mechanism extends into the palm, providing mechanical mounting for future tactile sensor arrays; (3) while joint-angle data is device-specific in theory, it is commonly fed directly to third-party robot hands in practice.

## Architecture

```
[20x RDC506018A Encoders] → [5x ADS1115 ADCs] → [STM32F103RCT6] → USB CDC → [Python PC Software]
                                                        ↓
                                                  [Debug LCD]
```

### Firmware (`firmware/`)
- **MCU:** STM32F103RCT6 (Cortex-M3, 72MHz), HAL-based
- **I2C1** (PB8/PB9, remapped): Thumb (0x48), Index (0x49), Middle (0x4A)
- **I2C2** (PB10/PB11): Ring (0x48), Pinky (0x49)
- **Data flow:** Poll 4 channels × 5 ADCs → pack 20 uint16 into frame → USB CDC TX
- **Frame format:** `[0xAA55][20×uint16][XOR checksum][0x0D]` = 44 bytes

### PC Software (`software/python/`)
- **Phase 1 — Skeleton Calibration (default):** Offline RGB photos + manual A4 4-corner marking (TL/TR/BR/BL) → homography to mm plane → MediaPipe landmarks → per-bone lengths + robust multi-image aggregation → `software/python/assets/skeletons/skeleton.json`
  - Detailed output: `offline_bone_mm_results.json`
  - Compact output: `software/python/assets/skeletons/skeleton.json` (default aggregate = median)
- **Phase 2 — Zero-Pose Calibration:** Flat-hand serial readings → `zero_pose.json` (angle offsets)
- **Phase 3 — Real-time Reconstruction:** Forward kinematics (DH parameters) + Vedo 3D viewer

### Joint Index Convention (0–19)
```
[0..3]   Thumb:  DIP, PIP, MCP_front, MCP_back
[4..7]   Index:  DIP, PIP, MCP_front, MCP_back
[8..11]  Middle: DIP, PIP, MCP_front, MCP_back
[12..15] Ring:   DIP, PIP, MCP_front, MCP_back
[16..19] Pinky:  DIP, PIP, MCP_front, MCP_back
```

Kinematics chain order (reversed from serial): MCP_back → MCP_front → PIP → DIP. DIP/PIP/MCP_front axes are parallel; MCP_back is orthogonal (abduction). Thumb kinematics is stubbed.

## Build & Run

### Firmware
```bash
cd firmware/dexslide_stm32
cmake --preset Debug          # configure (once), requires arm-none-eabi-gcc + CMake 3.22+ + Ninja
cmake --build build/Debug     # build
st-flash write build/Debug/dexslide_stm32.bin 0x08000000  # flash
```

### PC Software
```bash
cd software/python
pip install -r requirements.txt

python main.py calibrate-skeleton                              # Phase 1: offline A4 + manual marking
python main.py calibrate-skeleton --show-debug                 # Phase 1: per-image mm debug window
python main.py calibrate-skeleton --reuse-a4                   # Phase 1: reuse first A4 marking
python ../../firmware/dexslide_stm32/scripts/glove_calibrate.py --port /dev/ttyACM0
python main.py run --port /dev/ttyACM0
python main.py run --port /dev/ttyACM0                         # Phase 3: live 3D
python main.py raw --port /dev/ttyACM0                         # Debug: print raw values
```

## Firmware Development Notes

### CubeMX Workflow
The `.ioc` file is the CubeMX project. Re-generating from CubeMX overwrites code **outside** `USER CODE BEGIN/END` blocks. All hand-written firmware code must go inside those blocks. This applies to `main.c`, `stm32f1xx_hal_msp.c`, and `stm32f1xx_it.c`.

### Adding New Firmware Source Files
Add new `.c` files to the **top-level** `firmware/dexslide_stm32/CMakeLists.txt` under `target_sources`, not the CubeMX-generated `cmake/stm32cubemx/CMakeLists.txt`.

### USB CDC Transmit
`CDC_Transmit_FS()` is non-blocking — returns `USBD_BUSY` if a prior TX is still pending. The frame packing in `usb_comm.c` calls this.

### Memory Constraints
Linker script (`STM32F103XX_FLASH.ld`): heap 512 bytes, stack 1024 bytes. Increase if adding dynamic allocation or deep call chains.

### Key Globals (main.c)
`hi2c1`, `hi2c2` (I2C handles), `huart1` (debug USART), `hpcd_USB_FS` (USB PCD, declared in `usbd_conf.c`).

## Submodules

- Leap Motion camera support has been removed because the hardware path is deprecated.

## Testing

No formal test suite exists. Validation is manual:
- `python main.py raw --port /dev/ttyACM0` — verify raw serial frame values
- `python main.py calibrate-skeleton --show-debug` — verify A4标定、mm映射和骨长稳定性

## Key Files

- `firmware/Core/Src/ads1115.c` — ADS1115 I2C driver (single-shot, 860 SPS)
- `firmware/Core/Src/usb_comm.c` — Frame packing and USB CDC transmit
- `software/python/dexslide/serial_reader.py` — Frame parsing with sync/checksum
- `software/python/offline_a4_bone_mm.py` — Phase 1 离线A4标定 + mm骨长提取
- `software/python/plot_compare_skeletons.py` — skeleton叠加对比
- `software/python/demo_20dof_matplotlib.py` — 20DOF运动可视化
- `software/python/dexslide/kinematics/hand_model.py` — DH-parameter forward kinematics
- `docs/pinout.md` — Complete pin and address mapping

## Phase 1 Calibration Architecture (Offline A4 Pipeline)

The skeleton calibration uses an "image plane measurement → world plane mapping" paradigm:

- **Manual A4 points**: user clicks A4 corners in fixed order TL→TR→BR→BL
- **Homography**: map image pixels to A4 metric plane (mm)
- **Keypoints**: MediaPipe outputs 21 image landmarks, then transformed into mm
- **Bone lengths**: adjacent-joint Euclidean distances in mm
- **Aggregation**: robust multi-image statistics (median/mean) build compact `software/python/assets/skeletons/skeleton.json`
- **Artifacts**: keep both detailed per-image JSON and compact skeleton JSON

## Hardware Reference

Datasheets in `ChipFiles/`: STM32H743 (reference board — actual MCU is F103), ADS1255 (reference — actual ADC is ADS1115), SL6243A USB hub, RDC506018A encoder.

## Pin Note

I2C1 pins in `docs/pinout.md` (PB6/PB7) reflect default mapping; firmware CubeMX config uses **remapped** PB8/PB9. The CubeMX `.ioc` file is authoritative.
