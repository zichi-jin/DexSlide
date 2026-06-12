#include <System.h>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <utility>
#include <vector>

#include "imu_ring_buffer.hpp"

class RealsenseTopicSlamNode : public rclcpp::Node {
 public:
  RealsenseTopicSlamNode()
      : rclcpp::Node("realsense_topic_slam_node"),
        tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)) {
    vocab_ = this->declare_parameter<std::string>(
        "vocab",
        "/data/codes/DexSlide/umi_mono/external/ORB_SLAM3_fork/Vocabulary/ORBvoc.txt");
    settings_ = this->declare_parameter<std::string>(
        "settings", "/data/codes/DexSlide/umi_mono/config/RealSense_D435i_online.yaml");
    map_atlas_ = this->declare_parameter<std::string>("map_atlas", "");
    image_topic_ =
        this->declare_parameter<std::string>("image_topic", "/camera/camera/color/image_raw");
    accel_topic_ =
        this->declare_parameter<std::string>("accel_topic", "/camera/camera/accel/sample");
    gyro_topic_ = this->declare_parameter<std::string>("gyro_topic", "/camera/camera/gyro/sample");
    pose_topic_ = this->declare_parameter<std::string>("pose_topic", "/dexslide/slam/pose");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    camera_frame_ =
        this->declare_parameter<std::string>("camera_frame", "camera_color_optical_frame");
    max_lost_frames_ = this->declare_parameter<int>("max_lost_frames", 900);
    accel_gyro_pair_window_s_ =
        this->declare_parameter<double>("accel_gyro_pair_window_s", 0.020);
    activate_localization_mode_ =
        this->declare_parameter<bool>("activate_localization_mode", false);

    if (!is_regular_file(vocab_)) {
      RCLCPP_ERROR(get_logger(), "Vocabulary not found: %s", vocab_.c_str());
      rclcpp::shutdown();
      return;
    }
    if (!is_regular_file(settings_)) {
      RCLCPP_ERROR(get_logger(), "Setting file not found: %s", settings_.c_str());
      rclcpp::shutdown();
      return;
    }
    if (!map_atlas_.empty() && !is_regular_file(map_atlas_)) {
      RCLCPP_ERROR(get_logger(), "Load map file not found: %s", map_atlas_.c_str());
      rclcpp::shutdown();
      return;
    }

    RCLCPP_INFO(get_logger(), "Field                  | Value");
    RCLCPP_INFO(get_logger(), "-----------------------+------------------------------------");
    RCLCPP_INFO(get_logger(), "vocabulary             | %s", vocab_.c_str());
    RCLCPP_INFO(get_logger(), "setting                | %s", settings_.c_str());
    RCLCPP_INFO(
        get_logger(),
        "load_map               | %s",
        map_atlas_.empty() ? "(empty)" : map_atlas_.c_str());
    RCLCPP_INFO(get_logger(), "image_topic            | %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "accel_topic            | %s", accel_topic_.c_str());
    RCLCPP_INFO(get_logger(), "gyro_topic             | %s", gyro_topic_.c_str());
    RCLCPP_INFO(get_logger(), "pose_topic             | %s", pose_topic_.c_str());
    RCLCPP_INFO(get_logger(), "map_frame              | %s", map_frame_.c_str());
    RCLCPP_INFO(get_logger(), "camera_frame           | %s", camera_frame_.c_str());
    RCLCPP_INFO(get_logger(), "max_lost_frames        | %d", max_lost_frames_);
    RCLCPP_INFO(
        get_logger(), "accel_gyro_pair_window_s | %.3f", accel_gyro_pair_window_s_);

    try {
      slam_ = std::make_shared<ORB_SLAM3::System>(
          vocab_,
          settings_,
          ORB_SLAM3::System::IMU_MONOCULAR,
          false,
          map_atlas_,
          std::string());
      if (activate_localization_mode_) {
        slam_->ActivateLocalizationMode();
        RCLCPP_INFO(get_logger(), "Localization mode active, atlas read-only");
      } else {
        RCLCPP_INFO(get_logger(),
                    "Localization mode NOT activated (matches gopro_slam.cc); "
                    "local mapping thread will run but no atlas is saved to disk");
      }
      if (!map_atlas_.empty()) {
        RCLCPP_INFO(get_logger(), "Atlas loaded from: %s", map_atlas_.c_str());
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Failed to initialize ORB-SLAM3: %s", e.what());
      rclcpp::shutdown();
      return;
    }

    const rclcpp::SensorDataQoS qos;
    pose_publisher_ =
        this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, qos);

    // Separate callback groups so IMU samples can keep flowing into the ring
    // buffer while the image callback is busy running LocalizeMonocular. This
    // gives true real-time behavior: each frame is processed the moment it
    // arrives, with no polling-timer latency.
    image_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    imu_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions image_opts;
    image_opts.callback_group = image_cb_group_;
    rclcpp::SubscriptionOptions imu_opts;
    imu_opts.callback_group = imu_cb_group_;

    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, qos,
        std::bind(&RealsenseTopicSlamNode::on_image, this, std::placeholders::_1),
        image_opts);
    accel_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        accel_topic_, qos,
        std::bind(&RealsenseTopicSlamNode::on_accel, this, std::placeholders::_1),
        imu_opts);
    gyro_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        gyro_topic_, qos,
        std::bind(&RealsenseTopicSlamNode::on_gyro, this, std::placeholders::_1),
        imu_opts);
  }

  ~RealsenseTopicSlamNode() override {
    if (slam_) {
      slam_->Shutdown();
    }
  }

 private:
  struct MotionState {
    Eigen::Vector3f accel = Eigen::Vector3f::Zero();
    Eigen::Vector3f gyro = Eigen::Vector3f::Zero();
    double accel_t = 0.0;
    double gyro_t = 0.0;
    bool have_accel = false;
    bool have_gyro = false;
  };

  static bool is_regular_file(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_regular_file(path, ec);
  }

  static bool is_pose_valid(const Sophus::SE3f& T) {
    const Eigen::Vector3f t = T.translation();
    const Eigen::Quaternionf q = T.so3().unit_quaternion();
    return std::isfinite(t.x()) && std::isfinite(t.y()) && std::isfinite(t.z()) &&
           std::isfinite(q.x()) && std::isfinite(q.y()) && std::isfinite(q.z()) &&
           std::isfinite(q.w());
  }

  cv::Mat to_bgr_image(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    try {
      const auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
      return cv_ptr->image.clone();
    } catch (const cv_bridge::Exception&) {
      // Fall back to manual conversion based on the source encoding.
    }

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          1000,
          "Failed to decode image encoding '%s': %s",
          msg->encoding.c_str(),
          e.what());
      return {};
    }

    const cv::Mat& src = cv_ptr->image;
    if (src.empty()) {
      return {};
    }

    cv::Mat converted;
    try {
      if (msg->encoding == sensor_msgs::image_encodings::BGR8) {
        converted = src.clone();
      } else if (msg->encoding == sensor_msgs::image_encodings::RGB8) {
        cv::cvtColor(src, converted, cv::COLOR_RGB2BGR);
      } else if (msg->encoding == sensor_msgs::image_encodings::MONO8) {
        cv::cvtColor(src, converted, cv::COLOR_GRAY2BGR);
      } else if (msg->encoding == sensor_msgs::image_encodings::BGRA8) {
        cv::cvtColor(src, converted, cv::COLOR_BGRA2BGR);
      } else if (msg->encoding == sensor_msgs::image_encodings::RGBA8) {
        cv::cvtColor(src, converted, cv::COLOR_RGBA2BGR);
      } else if (src.channels() == 3) {
        converted = src.clone();
      } else if (src.channels() == 1) {
        cv::cvtColor(src, converted, cv::COLOR_GRAY2BGR);
      } else if (src.channels() == 4) {
        cv::cvtColor(src, converted, cv::COLOR_BGRA2BGR);
      } else {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "Unsupported image encoding '%s' (%d channels)",
            msg->encoding.c_str(),
            src.channels());
      }
    } catch (const cv::Exception& e) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000, "Image conversion failed: %s", e.what());
      return {};
    }

    return converted;
  }

  double session_relative(const builtin_interfaces::msg::Time& stamp) {
    const double abs_t = static_cast<double>(stamp.sec) +
                         static_cast<double>(stamp.nanosec) * 1e-9;
    // ORB-SLAM3's IMU pre-integration is sensitive to large timestamp magnitudes
    // (~1.7e9 s since Unix epoch) due to float precision. gopro_slam.cc and
    // realsense_online's playback mode both zero-align to the first sample.
    // Replicate that here using the first observed message timestamp as origin.
    {
      std::lock_guard<std::mutex> lock(session_mutex_);
      if (!session_initialized_) {
        session_origin_t_ = abs_t;
        session_initialized_ = true;
        RCLCPP_INFO(get_logger(),
                    "Session timestamp origin captured: %.6f s",
                    abs_t);
        return 0.0;
      }
    }
    return abs_t - session_origin_t_;
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    if (!slam_) {
      return;
    }

    cv::Mat image = to_bgr_image(msg);
    if (image.empty()) {
      return;
    }

    const double current_t = session_relative(msg->header.stamp);
    process_frame(image, current_t);
  }

  void process_frame(const cv::Mat& image, double current_t) {
    const std::vector<ImuSample> drained = imu_buf_.drain_until(current_t);
    std::vector<ORB_SLAM3::IMU::Point> v_imu;
    v_imu.reserve(drained.size());
    for (const ImuSample& s : drained) {
      if (s.t <= prev_t_ || s.t > current_t) {
        continue;
      }
      v_imu.emplace_back(s.a.x(), s.a.y(), s.a.z(), s.w.x(), s.w.y(), s.w.z(), s.t);
    }

    std::pair<Sophus::SE3f, bool> loc_result;
    try {
      loc_result = slam_->LocalizeMonocular(image, current_t, v_imu);
    } catch (const std::exception& e) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000, "LocalizeMonocular failed: %s", e.what());
      return;
    }

    const Sophus::SE3f& Tcw = loc_result.first;
    const bool ok = loc_result.second;
    const bool pose_valid = is_pose_valid(Tcw);

    if (!ok || !pose_valid) {
      ++consecutive_lost_;
      if (consecutive_lost_ > max_lost_frames_) {
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000, "Triggering soft reset (lost > max)");
        consecutive_lost_ = 0;
      }
      prev_t_ = current_t;
      return;
    }

    consecutive_lost_ = 0;
    publish_pose_and_tf(Tcw, current_t);
    prev_t_ = current_t;
  }

  void on_accel(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    const double t = session_relative(msg->header.stamp);
    const Eigen::Vector3f accel(
        static_cast<float>(msg->linear_acceleration.x),
        static_cast<float>(msg->linear_acceleration.y),
        static_cast<float>(msg->linear_acceleration.z));

    std::lock_guard<std::mutex> lock(motion_mutex_);
    motion_state_.accel = accel;
    motion_state_.accel_t = t;
    motion_state_.have_accel = true;
  }

  void on_gyro(const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
    const double t = session_relative(msg->header.stamp);
    const Eigen::Vector3f gyro(
        static_cast<float>(msg->angular_velocity.x),
        static_cast<float>(msg->angular_velocity.y),
        static_cast<float>(msg->angular_velocity.z));

    ImuSample sample;
    bool should_push = false;
    {
      std::lock_guard<std::mutex> lock(motion_mutex_);
      motion_state_.gyro = gyro;
      motion_state_.gyro_t = t;
      motion_state_.have_gyro = true;

      if (motion_state_.have_accel && motion_state_.have_gyro &&
          std::fabs(motion_state_.gyro_t - motion_state_.accel_t) < accel_gyro_pair_window_s_) {
        sample = ImuSample(motion_state_.accel, motion_state_.gyro, motion_state_.gyro_t);
        should_push = true;
      }
    }

    if (should_push) {
      imu_buf_.push(sample);
    }
  }

  void on_slam_timer() {
    // Legacy hook kept for binary/source compatibility. SLAM processing now
    // runs directly inside on_image() for minimal real-time latency.
  }

  void publish_pose_and_tf(const Sophus::SE3f& Tcw, double current_t) {
    const Sophus::SE3f Twc = Tcw.inverse();
    const Eigen::Vector3f t = Twc.translation();
    const Eigen::Quaternionf q = Twc.so3().unit_quaternion();
    const double abs_t = current_t + session_origin_t_;
    const rclcpp::Time stamp(static_cast<int64_t>(abs_t * 1e9));

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = map_frame_;
    pose_msg.pose.position.x = t.x();
    pose_msg.pose.position.y = t.y();
    pose_msg.pose.position.z = t.z();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    pose_publisher_->publish(pose_msg);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = camera_frame_;
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_msg.transform.rotation.w = q.w();
    tf_broadcaster_->sendTransform(tf_msg);
  }

  std::string vocab_;
  std::string settings_;
  std::string map_atlas_;
  std::string image_topic_;
  std::string accel_topic_;
  std::string gyro_topic_;
  std::string pose_topic_;
  std::string map_frame_;
  std::string camera_frame_;
  int max_lost_frames_ = 900;
  double accel_gyro_pair_window_s_ = 0.020;
  bool activate_localization_mode_ = false;

  std::shared_ptr<ORB_SLAM3::System> slam_;
  ImuRingBuffer<ImuSample, 256> imu_buf_;
  MotionState motion_state_;

  std::mutex session_mutex_;
  double session_origin_t_ = 0.0;
  bool session_initialized_ = false;

  std::mutex motion_mutex_;
  double prev_t_ = 0.0;
  int consecutive_lost_ = 0;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr accel_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr gyro_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::CallbackGroup::SharedPtr image_cb_group_;
  rclcpp::CallbackGroup::SharedPtr imu_cb_group_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RealsenseTopicSlamNode>();
  if (rclcpp::ok()) {
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  }
  rclcpp::shutdown();
  return 0;
}
