// Diagnostic: D435i image/IMU timestamp skew + long-window FPS
// Logs absolute timestamps so we can attribute the 32ms median skew
// observed in realsense_online startup check.

#include <librealsense2/rs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

namespace {

const char* DomainName(rs2_timestamp_domain d) {
  switch (d) {
    case RS2_TIMESTAMP_DOMAIN_HARDWARE_CLOCK: return "HW";
    case RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME:    return "SYS";
    case RS2_TIMESTAMP_DOMAIN_GLOBAL_TIME:    return "GLOBAL";
    default:                                  return "UNK";
  }
}

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t i = static_cast<size_t>(p * (v.size() - 1));
  return v[i];
}

}  // namespace

int main(int argc, char** argv) {
  const int duration_s = (argc > 1) ? std::atoi(argv[1]) : 15;

  try {
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0) {
      std::cerr << "No RealSense device" << std::endl;
      return 1;
    }
    rs2::device device = devices.front();

    // Enable global time on every supported sensor.
    for (rs2::sensor s : device.query_sensors()) {
      if (s.supports(RS2_OPTION_GLOBAL_TIME_ENABLED)) {
        s.set_option(RS2_OPTION_GLOBAL_TIME_ENABLED, 1.0f);
      }
    }

    // Report exposure setting.
    for (rs2::sensor s : device.query_sensors()) {
      if (std::string(s.get_info(RS2_CAMERA_INFO_NAME)).find("RGB") != std::string::npos) {
        const bool auto_exp = s.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)
                                  ? s.get_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE) > 0.5f
                                  : false;
        const float exp_us = s.supports(RS2_OPTION_EXPOSURE)
                                 ? s.get_option(RS2_OPTION_EXPOSURE)
                                 : -1.0f;
        std::cout << "RGB auto_exposure=" << (auto_exp ? "ON" : "OFF")
                  << " exposure_us=" << exp_us << std::endl;
      }
    }

    rs2::config cfg;
    cfg.enable_device(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
    cfg.enable_stream(RS2_STREAM_COLOR, 960, 540, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 200);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 200);

    std::mutex log_mtx;
    std::vector<double> color_ts_ms;     // GLOBAL time of color frames (ms)
    std::vector<double> color_arrival_ms;// steady-clock arrival of color frames (ms)
    std::vector<double> accel_ts_ms;
    std::vector<double> gyro_ts_ms;
    rs2_timestamp_domain color_dom = RS2_TIMESTAMP_DOMAIN_COUNT;
    rs2_timestamp_domain motion_dom = RS2_TIMESTAMP_DOMAIN_COUNT;

    const auto t0 = std::chrono::steady_clock::now();
    auto steady_ms = [&t0]() {
      const auto now = std::chrono::steady_clock::now();
      return std::chrono::duration<double, std::milli>(now - t0).count();
    };

    rs2::pipeline pipe(ctx);
    pipe.start(cfg, [&](const rs2::frame& f) {
      if (!f) return;
      if (auto motion = f.as<rs2::motion_frame>()) {
        const auto stream = motion.get_profile().stream_type();
        const double t_ms = motion.get_timestamp();
        motion_dom = motion.get_frame_timestamp_domain();
        std::lock_guard<std::mutex> lk(log_mtx);
        if (stream == RS2_STREAM_ACCEL) {
          accel_ts_ms.push_back(t_ms);
        } else if (stream == RS2_STREAM_GYRO) {
          gyro_ts_ms.push_back(t_ms);
        }
        return;
      }
      if (auto fs = f.as<rs2::frameset>()) {
        rs2::video_frame color = fs.get_color_frame();
        if (color) {
          const double t_ms = color.get_timestamp();
          color_dom = color.get_frame_timestamp_domain();
          std::lock_guard<std::mutex> lk(log_mtx);
          color_ts_ms.push_back(t_ms);
          color_arrival_ms.push_back(steady_ms());
        }
      }
    });

    std::this_thread::sleep_for(std::chrono::seconds(duration_s));
    pipe.stop();

    std::cout << std::fixed << std::setprecision(3);
    const double dur = static_cast<double>(duration_s);
    std::cout << "---- COUNTS over " << dur << "s ----\n";
    std::cout << "color: " << color_ts_ms.size() << " (" << color_ts_ms.size() / dur << " Hz)\n";
    std::cout << "accel: " << accel_ts_ms.size() << " (" << accel_ts_ms.size() / dur << " Hz)\n";
    std::cout << "gyro:  " << gyro_ts_ms.size() << " (" << gyro_ts_ms.size() / dur << " Hz)\n";
    std::cout << "color_domain=" << DomainName(color_dom) << ", motion_domain=" << DomainName(motion_dom) << "\n";

    // Inter-frame periods.
    auto periods = [](const std::vector<double>& v) {
      std::vector<double> p;
      for (size_t i = 1; i < v.size(); ++i) p.push_back(v[i] - v[i - 1]);
      return p;
    };

    auto print_periods = [](const char* name, const std::vector<double>& p) {
      if (p.empty()) {
        std::cout << name << " periods: <none>\n";
        return;
      }
      std::vector<double> sorted = p;
      std::sort(sorted.begin(), sorted.end());
      const double med = sorted[sorted.size() / 2];
      const double p99 = sorted[static_cast<size_t>(0.99 * (sorted.size() - 1))];
      const double max_p = sorted.back();
      std::cout << name << " period (ms): median=" << med << " p99=" << p99 << " max=" << max_p << "\n";
    };

    print_periods("color", periods(color_ts_ms));
    print_periods("accel", periods(accel_ts_ms));
    print_periods("gyro",  periods(gyro_ts_ms));

    // Color sensor timestamp vs steady-clock arrival drift (does color ts age behind arrival?).
    if (color_ts_ms.size() >= 5) {
      // Anchor first sample.
      const double dts0 = color_ts_ms.front();
      const double da0  = color_arrival_ms.front();
      std::vector<double> delays;
      for (size_t i = 0; i < color_ts_ms.size(); ++i) {
        const double rel_ts = color_ts_ms[i] - dts0;
        const double rel_arr = color_arrival_ms[i] - da0;
        delays.push_back(rel_arr - rel_ts);   // positive = arrival later than its timestamp
      }
      std::cout << "color arrival - timestamp (ms): median=" << Median(delays)
                << " p99=" << Percentile(delays, 0.99)
                << " min=" << *std::min_element(delays.begin(), delays.end())
                << " max=" << *std::max_element(delays.begin(), delays.end()) << "\n";
    }

    // Per-color-frame skew vs nearest IMU sample (true skew, not "latest IMU").
    if (!color_ts_ms.empty() && !accel_ts_ms.empty()) {
      std::vector<double> skew_nearest_ms;
      size_t j = 0;
      for (size_t i = 0; i < color_ts_ms.size(); ++i) {
        const double ct = color_ts_ms[i];
        while (j + 1 < accel_ts_ms.size() && accel_ts_ms[j + 1] <= ct) ++j;
        // pick nearest between j and j+1
        double best = std::fabs(accel_ts_ms[j] - ct);
        if (j + 1 < accel_ts_ms.size()) {
          best = std::min(best, std::fabs(accel_ts_ms[j + 1] - ct));
        }
        skew_nearest_ms.push_back(best);
      }
      std::cout << "|color - nearest accel| (ms): median=" << Median(skew_nearest_ms)
                << " p99=" << Percentile(skew_nearest_ms, 0.99) << "\n";
    }

    // Mimic realsense_online "latest IMU at image arrival" skew.
    // Here we use the rule "IMU sample with largest accel_ts <= color_ts" as latest.
    if (!color_ts_ms.empty() && !accel_ts_ms.empty()) {
      std::vector<double> latest_skew_ms;
      size_t j = 0;
      for (size_t i = 0; i < color_ts_ms.size(); ++i) {
        const double ct = color_ts_ms[i];
        while (j + 1 < accel_ts_ms.size() && accel_ts_ms[j + 1] <= ct) ++j;
        latest_skew_ms.push_back(std::fabs(ct - accel_ts_ms[j]));
      }
      std::cout << "|color - latest_imu<=color| (ms): median=" << Median(latest_skew_ms)
                << " p99=" << Percentile(latest_skew_ms, 0.99) << "\n";
    }

    return 0;
  } catch (const rs2::error& e) {
    std::cerr << "RS error: " << e.what() << std::endl;
    return 1;
  }
}
