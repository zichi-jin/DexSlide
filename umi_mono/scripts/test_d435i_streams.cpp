// TASK-005 D435i librealsense2 smoke test for online tracking mode.
// Tracker: /home/jzq/MyJob/DexSlide/umi_mono/docs/online_tracking_implementation.md
// Opens color, accel, and gyro streams and measures observed frame rates.
// Enables global time on supported sensors before starting the pipeline.
// Prints FPS plus timestamp domain and exits non-zero on failure.

#include <librealsense2/rs.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

struct Counts {
  std::atomic<int> color{0};
  std::atomic<int> accel{0};
  std::atomic<int> gyro{0};
};

const char* TimestampDomainName(rs2_timestamp_domain domain) {
  switch (domain) {
    case RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK:
      return "HARDWARE_CLOCK";
    case RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME:
      return "SYSTEM_TIME";
    case RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME:
      return "GLOBAL_TIME";
    default:
      return "UNKNOWN";
  }
}

bool InRange(double value, double min_value, double max_value) {
  return value >= min_value && value <= max_value;
}

void EnableGlobalTimeIfSupported(const rs2::device& device) {
  for (rs2::sensor sensor : device.query_sensors()) {
    if (sensor.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
      sensor.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
    }
  }
}

}  // namespace

int main() {
  try {
    Counts counts;
    std::atomic<rs2_timestamp_domain> motion_domain{RS2_TIMESTAMP_DOMAIN_COUNT};
    std::atomic<rs2_timestamp_domain> color_domain{RS2_TIMESTAMP_DOMAIN_COUNT};

    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
      std::cerr << "No RealSense device detected" << std::endl;
      return 1;
    }

    rs2::device device = devices.front();
    EnableGlobalTimeIfSupported(device);

    rs2::config cfg;
    cfg.enable_device(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg.enable_stream(RS2_STREAM_COLOR, 960, 540, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 200);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

    rs2::pipeline pipe(ctx);
    pipe.start(cfg, [&](const rs2::frame& frame) {
      if (!frame) {
        return;
      }

      if (auto motion = frame.as<rs2::motion_frame>()) {
        const rs2_stream stream = motion.get_profile().stream_type();
        motion_domain.store(motion.get_frame_timestamp_domain(), std::memory_order_relaxed);
        if (stream == RS2_STREAM_ACCEL) {
          counts.accel.fetch_add(1, std::memory_order_relaxed);
        } else if (stream == RS2_STREAM_GYRO) {
          counts.gyro.fetch_add(1, std::memory_order_relaxed);
        }
        return;
      }

      if (auto frameset = frame.as<rs2::frameset>()) {
        rs2::video_frame color = frameset.get_color_frame();
        if (color) {
          counts.color.fetch_add(1, std::memory_order_relaxed);
          color_domain.store(color.get_frame_timestamp_domain(), std::memory_order_relaxed);
        }
      }
    });

    std::this_thread::sleep_for(std::chrono::seconds(5));

    pipe.stop();

    constexpr double kDurationSeconds = 5.0;
    const double color_fps = static_cast<double>(counts.color.load()) / kDurationSeconds;
    const double accel_fps = static_cast<double>(counts.accel.load()) / kDurationSeconds;
    const double gyro_fps = static_cast<double>(counts.gyro.load()) / kDurationSeconds;

    rs2_timestamp_domain chosen_domain = color_domain.load(std::memory_order_relaxed);
    if (motion_domain.load(std::memory_order_relaxed) != RS2_TIMESTAMP_DOMAIN_COUNT) {
      chosen_domain = motion_domain.load(std::memory_order_relaxed);
    }

    std::ostringstream line;
    line << std::fixed << std::setprecision(1)
         << "color FPS=" << color_fps
         << ", accel FPS=" << accel_fps
         << ", gyro FPS=" << gyro_fps
         << ", timestamp_domain=" << TimestampDomainName(chosen_domain);
    std::cout << line.str() << std::endl;

    const bool pass = InRange(color_fps, 27.0, 33.0) &&
                      InRange(accel_fps, 180.0, 220.0) &&
                      InRange(gyro_fps, 180.0, 220.0);

    std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 1;
  } catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
