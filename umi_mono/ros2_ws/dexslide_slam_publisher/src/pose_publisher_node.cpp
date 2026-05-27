#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <zmq.hpp>

#include <cstdio>
#include <memory>
#include <string>

class PosePublisherNode : public rclcpp::Node {
 public:
  PosePublisherNode()
      : rclcpp::Node("pose_publisher_node"),
        tf_broadcaster_(std::make_unique<tf2_ros::TransformBroadcaster>(*this)),
        zmq_context_(1),
        zmq_socket_(zmq_context_, ZMQ_SUB) {
    const std::string zmq_endpoint =
        this->declare_parameter<std::string>("zmq_endpoint", "tcp://127.0.0.1:5555");
    pose_topic_ = this->declare_parameter<std::string>("pose_topic", "/dexslide/slam/pose");
    world_frame_ = this->declare_parameter<std::string>("world_frame", "world");
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    camera_frame_ =
        this->declare_parameter<std::string>("camera_frame", "camera_color_optical_frame");
    const int recv_timeout_ms = this->declare_parameter<int>("recv_timeout_ms", 100);

    rclcpp::SensorDataQoS qos;
    qos.keep_last(10);
    pose_publisher_ =
        this->create_publisher<geometry_msgs::msg::PoseStamped>(pose_topic_, qos);

    zmq_socket_.set(zmq::sockopt::subscribe, "");
    zmq_socket_.set(zmq::sockopt::rcvtimeo, recv_timeout_ms);
    zmq_socket_.connect(zmq_endpoint);

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1), std::bind(&PosePublisherNode::poll_and_publish, this));

    RCLCPP_INFO(this->get_logger(), "Listening on %s, publishing %s",
                zmq_endpoint.c_str(), pose_topic_.c_str());
  }

 private:
  void poll_and_publish() {
    zmq::message_t msg;
    const zmq::recv_result_t recv_result = zmq_socket_.recv(msg, zmq::recv_flags::dontwait);
    if (!recv_result || *recv_result == 0) {
      return;
    }

    const std::string buffer(static_cast<const char*>(msg.data()), msg.size());

    double t = 0.0;
    double tx = 0.0;
    double ty = 0.0;
    double tz = 0.0;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 0.0;
    const int matched = std::sscanf(
        buffer.c_str(),
        "{\"t\":%lf,\"tx\":%lf,\"ty\":%lf,\"tz\":%lf,\"qx\":%lf,\"qy\":%lf,\"qz\":%lf,\"qw\":%lf}",
        &t, &tx, &ty, &tz, &qx, &qy, &qz, &qw);
    if (matched != 8) {
      RCLCPP_DEBUG(this->get_logger(), "Skipping malformed ZMQ pose: %s", buffer.c_str());
      return;
    }

    const rclcpp::Time stamp(static_cast<int64_t>(t * 1e9));

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = map_frame_;
    pose_msg.pose.position.x = tx;
    pose_msg.pose.position.y = ty;
    pose_msg.pose.position.z = tz;
    pose_msg.pose.orientation.x = qx;
    pose_msg.pose.orientation.y = qy;
    pose_msg.pose.orientation.z = qz;
    pose_msg.pose.orientation.w = qw;
    pose_publisher_->publish(pose_msg);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = stamp;
    tf_msg.header.frame_id = map_frame_;
    tf_msg.child_frame_id = camera_frame_;
    tf_msg.transform.translation.x = tx;
    tf_msg.transform.translation.y = ty;
    tf_msg.transform.translation.z = tz;
    tf_msg.transform.rotation.x = qx;
    tf_msg.transform.rotation.y = qy;
    tf_msg.transform.rotation.z = qz;
    tf_msg.transform.rotation.w = qw;
    tf_broadcaster_->sendTransform(tf_msg);
  }

  std::string pose_topic_;
  std::string world_frame_;
  std::string map_frame_;
  std::string camera_frame_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  zmq::context_t zmq_context_;
  zmq::socket_t zmq_socket_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PosePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
