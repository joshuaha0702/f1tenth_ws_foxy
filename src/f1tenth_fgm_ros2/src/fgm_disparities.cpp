#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "visualization_msgs/msg/marker.hpp"

// [하드웨어 호환성] 실제 차량 구동을 위한 Ackermann 메시지 헤더 추가
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <ctime> 
#include <iostream>
#include <memory>
#include <string>
#include <vector>

float translate(float value, float leftMin, float leftMax, float rightMin, float rightMax) {
    float leftSpan = leftMax - leftMin;
    float rightSpan = rightMax - rightMin;
    float valueScaled = (value - leftMin) / leftSpan;
    return rightMin + (valueScaled * rightSpan);
}


class FGMNode : public rclcpp::Node {
public:
    FGMNode() : Node("fgm_disparities") {
        std::string robot_name = this->declare_parameter("robot_name", "car1");

        std::string drive_topic = this->declare_parameter("drive_topic", "drive");
        std::string lidar_topic = this->declare_parameter("lidar_topic", "scan");
        std::string lidar_pub_topic = this->declare_parameter("lidar_pub_topic", "disparity_lidar");
        std::string arrow_marker_topic = this->declare_parameter("arrow_marker_topic", "direction_marker");

        car_length = this->declare_parameter("car_length", 0.33);
        car_width = this->declare_parameter("car_width", 0.20);
        max_speed = this->declare_parameter("max_speed", 2.0);
        min_speed = this->declare_parameter("min_speed", 0.5);
        max_steering_angle = this->declare_parameter("max_steering_angle", 0.4189);
        disp_offset = this->declare_parameter("disp_offset", 0.1);
        carWidth_tolerance = this->declare_parameter("carWidth_tolerance", 0.4);
        max_distance = this->declare_parameter("max_distance", 10.0);

        // [하드웨어 호환성] 실제 차량(VESC 등)은 AckermannDriveStamped 타입을 사용하므로 변경함
        drive_pub = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 10);
        lidar_pub = this->create_publisher<sensor_msgs::msg::LaserScan>(lidar_pub_topic, 10);
        arrow_marker_pub = this->create_publisher<visualization_msgs::msg::Marker>(arrow_marker_topic, 10);

        lidar_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidar_topic, 
            rclcpp::SensorDataQoS(),
            std::bind(&FGMNode::lidar_callback, this, std::placeholders::_1)
        );

        chord_length = (car_width / 2) + carWidth_tolerance;

        // [하드웨어 호환성] AckermannDriveStamped 메시지 객체 초기화
        drive_cmd.drive.steering_angle_velocity = 0.0;
        drive_cmd.drive.acceleration = 0.0;
        drive_cmd.drive.jerk = 0.0;
        drive_cmd.drive.speed = 0.0;
        drive_cmd.drive.steering_angle = 0.0;

        std::string laser_frame_id = robot_name + "/laser";

        disparity_lidar.ranges.resize(1081);
        disparity_lidar.intensities.resize(1081);
        disparity_lidar.header.frame_id = laser_frame_id;
        disparity_lidar.angle_min = -2.35619;
        disparity_lidar.angle_max = 2.35619;
        disparity_lidar.angle_increment = 0.00436331;
        disparity_lidar.time_increment = 0.0;
        disparity_lidar.scan_time = 0.0;
        disparity_lidar.range_min = 0.1;
        disparity_lidar.range_max = max_distance;

        direction_arrow.pose.position.x = 0.0;
        direction_arrow.pose.position.y = 0.0;
        direction_arrow.header.frame_id = laser_frame_id;
        direction_arrow.type = visualization_msgs::msg::Marker::ARROW;
        direction_arrow.id = 0;
        direction_arrow.ns = "car_forward_direction";
        direction_arrow.scale.x = 1.0;
        direction_arrow.scale.y = 0.1;
        direction_arrow.scale.z = 0.1;
        direction_arrow.color.r = 1.0;
        direction_arrow.color.a = 1.0;
    }

private:
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        auto start = this->now();
        int size = msg->ranges.size();
        
        // 입력 데이터 크기에 맞춰 행렬 리사이즈 및 초기화
        if (range_matrix.rows() != size) {
            range_matrix.resize(size, 3);
        }
        range_matrix.setZero();

        float starting_angle = msg->angle_min;
        float angle_increment = msg->angle_increment;
        
        // 각도 기반 인덱스 계산 (-90도 ~ +90도 범위)
        int start_index = static_cast<int>((-1.5708f - starting_angle) / angle_increment);
        int stop_index = static_cast<int>((1.5708f - starting_angle) / angle_increment);
        
        // 인덱스 안전 범위 제한
        start_index = std::max(0, std::min(start_index, size - 1));
        stop_index = std::max(0, std::min(stop_index, size - 2)); // i+1 접근을 위해 -2

        float angle = starting_angle + start_index * angle_increment;
        float angle_stopper = 0;
        int i = start_index;
        double right_hs, left_hs;
        after_LD = false;

        while(i <= stop_index) {
            right_hs = msg->ranges[i+1];
            left_hs = msg->ranges[i];

            // 유효하지 않은 데이터 처리 (Inf, NaN)
            if (!std::isfinite(left_hs)) left_hs = max_distance;
            if (!std::isfinite(right_hs)) right_hs = max_distance;

            if(right_hs - left_hs > 0.15) {
                range_matrix(i,2) = 1;
                range_matrix(i,0) = left_hs;
                
                float asin_arg = chord_length / (2.0f * left_hs);
                if (asin_arg > 1.0f) asin_arg = 1.0f;
                car_arc_length = 2.0f * left_hs * std::asin(asin_arg);
                
                alpha = (car_arc_length) / left_hs;
                angle_stopper = 0;
                while(angle_stopper <= alpha && i + 1 < size) {
                    range_matrix(i+1,0) = left_hs;
                    range_matrix(i,1) = angle;
                    angle_stopper += angle_increment;
                    angle += angle_increment;
                    i++;
                    if (i < size) range_matrix(i, 2) = 9;
                }
            }
            else if(left_hs - right_hs > 0.15) {
                if (i + 1 < size) range_matrix(i+1,2) = 2;
                
                float asin_arg = chord_length / (2.0f * right_hs);
                if (asin_arg > 1.0f) asin_arg = 1.0f;
                car_arc_length = 2.0f * right_hs * std::asin(asin_arg);

                alpha = (car_arc_length) / right_hs;
                angle_stopper = 0;
                int j = i;
                while(angle_stopper <= alpha && j >= 0) {
                    range_matrix(j,2) = 9;
                    range_matrix(j,0) = right_hs;
                    angle_stopper += angle_increment;
                    j--;
                }
                range_matrix(i,1) = angle;
                angle += angle_increment;
                i++;
                after_LD = true;
            }
            else {
                range_matrix(i,0) = left_hs;
                range_matrix(i,1) = angle;
                if(after_LD) {
                    range_matrix(i,2) = 2;
                    after_LD = false;
                }
                else {
                    range_matrix(i,2) = 0;
                }
                angle += angle_increment;
                i++;
            }
        }
        
        // 가상 라이다 데이터 준비
        if (disparity_lidar.ranges.size() != static_cast<size_t>(size)) {
            disparity_lidar.ranges.resize(size);
            disparity_lidar.intensities.resize(size);
        }
        disparity_lidar.angle_min = msg->angle_min;
        disparity_lidar.angle_max = msg->angle_max;
        disparity_lidar.angle_increment = msg->angle_increment;

        for (int k = 0; k < size; ++k) {
            disparity_lidar.ranges[k] = range_matrix(k,0);
            disparity_lidar.intensities[k] = range_matrix(k,1);
        }

        furthest_distance = range_matrix.col(0).maxCoeff(&furthest_distance_index);
        float roll = 0, pitch = 0, yaw = starting_angle + furthest_distance_index*angle_increment; 
        Eigen::Quaternionf q;
        q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        
        direction_arrow.pose.orientation.x = q.x();
        direction_arrow.pose.orientation.y = q.y();
        direction_arrow.pose.orientation.z = q.z();
        direction_arrow.pose.orientation.w = q.w();

        float steering_angle = starting_angle + furthest_distance_index*angle_increment;
        if(steering_angle > max_steering_angle) steering_angle = max_steering_angle;
        else if(steering_angle < -max_steering_angle) steering_angle = -max_steering_angle;

        if (!std::isfinite(furthest_distance) || furthest_distance > max_distance) furthest_distance = max_distance;
        
        speed = translate(furthest_distance, 0.0, max_distance, min_speed, max_speed);
        if(steering_angle <= 0.1 && steering_angle >= -0.1) speed = speed * 0.95;
        else if(steering_angle <= 0.3 && steering_angle >= -0.3) speed = speed * 0.65;
        else speed = speed * 0.35;
        
        // [하드웨어 호환성] Ackermann 제어 명령 할당 (Twist의 linear.x, angular.z 대신 직접 필드 사용)
        drive_cmd.drive.steering_angle = steering_angle;
        drive_cmd.drive.speed = speed;

        auto current_time = msg->header.stamp;
        disparity_lidar.header.stamp = current_time;
        direction_arrow.header.stamp = current_time;
        // [하드웨어 호환성] 제어 동기화를 위해 헤더 타임스탬프 설정 필수
        drive_cmd.header.stamp = current_time;
        
        arrow_marker_pub->publish(direction_arrow);
        drive_pub->publish(drive_cmd);
        lidar_pub->publish(disparity_lidar);
    }

    // [하드웨어 호환성] 퍼블리셔 타입을 Ackermann 타입으로 변경
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr lidar_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr arrow_marker_pub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub;

    double max_speed, min_speed, max_steering_angle, max_distance;
    float speed, car_length, car_width, alpha, car_arc_length, disp_offset, chord_length, carWidth_tolerance;
    bool after_LD = false;

    Eigen::MatrixXd range_matrix = Eigen::MatrixXd::Zero(1081, 3);
    
    sensor_msgs::msg::LaserScan disparity_lidar;
    visualization_msgs::msg::Marker direction_arrow;
    // [하드웨어 호환성] 메시지 객체 타입을 Ackermann 타입으로 변경
    ackermann_msgs::msg::AckermannDriveStamped drive_cmd;
    
    Eigen::MatrixXd::Index furthest_distance_index;
    float furthest_distance;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<FGMNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
