#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

class Float64ArraySubscriber : public rclcpp::Node
{
public:
  Float64ArraySubscriber()
      : Node("float64_array_subscriber")
  {
    subscription_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "omni_data", 10, std::bind(&Float64ArraySubscriber::callback, this, std::placeholders::_1));
  }

private:
  void callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) const
  {
    std::cout << "I heard: ";
    for (const auto& value : msg->data)
      std::cout << value << " ";
    std::cout << std::endl;
  }

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Float64ArraySubscriber>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
