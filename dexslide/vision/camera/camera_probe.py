"""实时探测双目相机实际接受的分辨率和帧率。"""

import argparse
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe a side-by-side stereo camera mode and measure delivered FPS."
    )
    parser.add_argument("--index", type=int, default=0, help="camera device index")
    parser.add_argument("--width", type=int, default=1920, help="requested camera frame width")
    parser.add_argument("--height", type=int, default=1192, help="requested camera frame height")
    parser.add_argument("--fps", type=float, default=60.0, help="requested camera FPS")
    return parser.parse_args()


def read_property(camera, prop):
    value = camera.get(prop)
    return value if value > 0 else None


def fourcc_to_text(value):
    if value is None:
        return "unknown"
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def print_mode_report(camera, args):
    actual_width = read_property(camera, cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = read_property(camera, cv2.CAP_PROP_FRAME_HEIGHT)
    driver_fps = read_property(camera, cv2.CAP_PROP_FPS)
    actual_fourcc = fourcc_to_text(read_property(camera, cv2.CAP_PROP_FOURCC))

    print("=== Requested mode ===")
    print(f"resolution: {args.width} x {args.height}")
    print(f"fps: {args.fps:.2f}")
    print("=== Driver-reported mode ===")
    print(f"resolution: {actual_width or 'unknown'} x {actual_height or 'unknown'}")
    print(f"fps: {driver_fps or 'unknown'}")
    print(f"fourcc: {actual_fourcc}")


def put_text(frame, text, line):
    cv2.putText(
        frame,
        text,
        (12, 28 + line * 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    camera = cv2.VideoCapture(args.index)

    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.index}. Try another --index value."
        )

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)

    print_mode_report(camera, args)
    print("Press q or Esc to quit.")

    measured_fps = 0.0
    frame_count = 0
    measure_start = time.perf_counter()

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read a frame.")
                break

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - measure_start
            if elapsed >= 1.0:
                measured_fps = frame_count / elapsed
                print(
                    f"live: {frame.shape[1]} x {frame.shape[0]}, "
                    f"measured_fps: {measured_fps:.2f}"
                )
                frame_count = 0
                measure_start = now

            frame_height, frame_width = frame.shape[:2]
            display = frame.copy()
            put_text(display, f"camera: {frame_width} x {frame_height}", 0)
            put_text(display, f"measured FPS: {measured_fps:.2f}", 1)

            cv2.imshow("camera", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
