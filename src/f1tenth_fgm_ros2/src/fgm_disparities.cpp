#include "rclcpp/rclcpp.hpp"                      // ros/ros.h 대체
#include "nav_msgs/msg/odometry.hpp"             // nav_msgs/Odometry.h 대체
#include "sensor_msgs/msg/laser_scan.hpp"         // sensor_msgs/LaserScan.h 대체
#include "visualization_msgs/msg/marker.hpp"      // visualization_msgs/Marker.h 대체

// #include "ackermann_msgs/msg/ackermann_drive_stamped.hpp" // ackermann_msgs/AckermannDriveStamped.h 대체
// #include "ackermann_msgs/msg/ackermann_drive.hpp"         // ackermann_msgs/AckermannDrive.h 대체

#include "geometry_msgs/msg/twist.hpp"

// 표준 C++ 및 라이브러리 헤더 (기존과 동일하거나 추가됨)
#include <Eigen/Dense>      // Eigen3 경로가 표준화됨 (CMake에서 Eigen3::Eigen 연결 시)
#include <cmath>
#include <chrono>
#include <ctime> 
#include <iostream>
#include <memory>           // ROS 2 스마트 포인터 사용을 위해 필수
#include <string>
#include <vector>

// 이 함수는 그대로 클래스 위에 두세요.
float translate(float value, float leftMin, float leftMax, float rightMin, float rightMax) {
    float leftSpan = leftMax - leftMin;
    float rightSpan = rightMax - rightMin;
    float valueScaled = (value - leftMin) / leftSpan;
    return rightMin + (valueScaled * rightSpan);
}


class FGMNode : public rclcpp::Node {
public:
    // 생성자: 반드시 public!
    FGMNode() : Node("fgm_disparities") {
        // 1. 파라미터 선언 및 가져오기 (이름, 기본값)
        // ROS 2에서는 선언과 동시에 값을 리턴받을 수 있어 한 줄로 끝낼 수 있습니다.
        // [추가됨] 터미널에서 로봇 이름을 파라미터로 받습니다 (기본값: car1)
        std::string robot_name = this->declare_parameter("robot_name", "car1");

        // [수정됨] 토픽 이름 앞에 로봇 이름을 붙여서 동적으로 생성합니다.
        std::string drive_topic = this->declare_parameter("drive_topic", "/" + robot_name + "/drive");
        std::string lidar_topic = this->declare_parameter("lidar_topic", "/" + robot_name + "/scan");
        std::string lidar_pub_topic = this->declare_parameter("lidar_pub_topic", "/" + robot_name + "/disparity_lidar");
        std::string arrow_marker_topic = this->declare_parameter("arrow_marker_topic", "/" + robot_name + "/direction_marker");

        car_length = this->declare_parameter("car_length", 0.33);
        car_width = this->declare_parameter("car_width", 0.20);
        max_speed = this->declare_parameter("max_speed", 2.0);
        min_speed = this->declare_parameter("min_speed", 0.5);
        max_steering_angle = this->declare_parameter("max_steering_angle", 0.4189);
        disp_offset = this->declare_parameter("disp_offset", 0.1);
        carWidth_tolerance = this->declare_parameter("carWidth_tolerance", 0.4);

        // 2. 퍼블리셔 설정 (this->create_publisher)
        drive_pub = this->create_publisher<geometry_msgs::msg::Twist>(drive_topic, 10);
        lidar_pub = this->create_publisher<sensor_msgs::msg::LaserScan>(lidar_pub_topic, 10);
        arrow_marker_pub = this->create_publisher<visualization_msgs::msg::Marker>(arrow_marker_topic, 10);

        // 3. 서브스크라이버 설정 (this->create_subscription)
        lidar_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            lidar_topic, 
            rclcpp::SensorDataQoS(), // 센서 데이터용 QoS 설정
            std::bind(&FGMNode::lidar_callback, this, std::placeholders::_1)
    );

    // 4. 초기 계산 로직
    chord_length = (car_width / 2) + carWidth_tolerance;

    // 5. AckermannDriveStamped 메시지 초기화
    // drive_cmd.drive.steering_angle_velocity = 0.0;
    // drive_cmd.drive.acceleration = 0.0;
    // drive_cmd.drive.jerk = 0.0;

    // 5. Twist 메시지 초기화 (Twist는 기본적으로 모든 값이 0.0으로 시작합니다)
    drive_cmd.linear.x = 0.0;
    drive_cmd.linear.y = 0.0;
    drive_cmd.linear.z = 0.0;
    drive_cmd.angular.x = 0.0;
    drive_cmd.angular.y = 0.0;
    drive_cmd.angular.z = 0.0;

    // [수정됨] TF 충돌을 막기 위해 프레임 이름에도 로봇 이름을 붙여줍니다 (예: car1/laser)
    std::string laser_frame_id = robot_name + "/laser";

    // 6. 가상 라이다 시각화 설정
    disparity_lidar.ranges.resize(1081);
    disparity_lidar.intensities.resize(1081);
    disparity_lidar.header.frame_id = "laser";
    disparity_lidar.angle_min = -2.35619;
    disparity_lidar.angle_max = 2.35619;
    disparity_lidar.angle_increment = 0.00436331;
    disparity_lidar.time_increment = 0.0;
    disparity_lidar.scan_time = 0.0;
    disparity_lidar.range_min = 0.1;
    disparity_lidar.range_max = 10.0;

    // 7. 화살표 마커 시각화 설정
    direction_arrow.pose.position.x = 0.0;
    direction_arrow.pose.position.y = 0.0;
    direction_arrow.header.frame_id = "laser";
    direction_arrow.type = visualization_msgs::msg::Marker::ARROW; // 0 대신 명확한 상수를 씀
    direction_arrow.id = 0;
    direction_arrow.ns = "car_forward_direction";
    direction_arrow.scale.x = 1.0; // 길이
    direction_arrow.scale.y = 0.1; // 폭
    direction_arrow.scale.z = 0.1; // 두께 (ROS 2는 z가 0이면 안 보일 수 있음)
    direction_arrow.color.r = 1.0; // 색상 추가 (안 하면 투명하게 나옴)
    direction_arrow.color.a = 1.0;
}

private:
    // 콜백 함수: 외부에서 부를 일 없으니 private이 좋습니다!
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        // 여기에 기존 FGM 계산 알고리즘 로직을 넣으세요.
        auto start = this->now();
        /*
            header: 
            seq: 57
            stamp: 1.792000000
            frame_id: laser
            angle_min: -2.35619
            angle_max: 2.35619
            angle_increment: 0.00436331
            time_increment: 0
            scan_time: 0
            range_min: 0.1
            range_max: 10
            ranges: [1081]
            intensities: [1081]
         
        Lidar perspective on the car
                                    FRONT
                                     | 0
                                     | x axis
                      0.959          |
                                    /|\ car forward direction
                                     |
               1.57 _________________|___________________ -1.57   y axis
                                     | 
                                     |
                      2.356 (135)    |      -2.356 (-135)
                                     |
                                3.14 | -3.14
                                    BACK
        */

        // Firstly, we need to reduce the fov of the lidar data to -90 to 90 degree
        // index: 180 angle: -1.5708
        // index: 900 angle: 1.57078
        // std::cout << msg << std::endl;
        int size = msg->ranges.size(); //1081 
        float starting_angle = msg->angle_min;
        float angle_increment = msg->angle_increment;
        float angle = starting_angle + 180*angle_increment;
        float angle_stopper = 0;
        int stop_index = 900;
        int i = 180;
        double right_hs, left_hs;
        while(i <= stop_index) { // index 900 is included
            // checking for disparities
            right_hs = msg->ranges[i+1];
            left_hs = msg->ranges[i];
            if(right_hs - left_hs > 0.15) {
                //std::cout << "detected right disparity" << std::endl;
                range_matrix(i,2) = 1;
                range_matrix(i,0) = left_hs;
                car_arc_length = 2 * left_hs * std::asin(chord_length/(2*left_hs));
                alpha = (car_arc_length) / left_hs;
                //std::cout << "car_arc_length is: " << car_arc_length << "left_hs is" << left_hs << std::endl;
                //std::cout << "detected right disparity and alpha is: " << alpha << std::endl;
                angle_stopper = 0; // 0 radian
                while(angle_stopper <= alpha && i + 1 < 1081) {

                    range_matrix(i+1,0) = left_hs;
                    range_matrix(i,1) = angle;
                    angle_stopper += angle_increment;
                    angle += angle_increment;
                    //std::cout << "i'th: " << i << " range_matrix(i,0) is: " << range_matrix(i,0) << " angle_stopper is: " << angle_stopper << std::endl;
                    i++; // i move for every filtering the amount angle alpha
                    range_matrix(i,2) = 9;
                }
                
            }
            else if(left_hs - right_hs > 0.15) {
                //std::cout << "detected left disparity" << std::endl;
                range_matrix(i+1,2) = 2;
                //std::cout << "car_arc_length is: " << car_arc_length << "left_hs is" << left_hs << std::endl;
                //std::cout << "detected left disparity and alpha is: " << alpha << std::endl;
                car_arc_length = 2 * right_hs * std::asin(chord_length/(2*right_hs));
                alpha = (car_arc_length) / right_hs; // right_hs as the radius of the big circle
                angle_stopper = 0; // 0 radian
                int j = i; // we need j because we are going reverse assigning
                while(angle_stopper <= alpha && j >= 0) {
                    range_matrix(j,2) = 9;
                    range_matrix(j,0) = right_hs;
                    angle_stopper += angle_increment;
                    //std::cout << "j'th: " << j << " range_matrix(i,0) is: " << range_matrix(j,0) << " angle_stopper is: " << angle_stopper << std::endl;
                    j--;
                }
                range_matrix(i,1) = angle;
                angle += angle_increment;
                i++; // i stays the same while j goes for reverse filtering
                after_LD = true;
            }
            else {
                range_matrix(i,0) = msg->ranges[i];
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
        
        //visualization
        i = 0;
        while(i < size) {
            disparity_lidar.ranges[i] = range_matrix(i,0);
            disparity_lidar.intensities[i] = range_matrix(i,1);
            //std::cout << "i'th: " << i << " range data: " << range_matrix(i,0) << " angle: " << range_matrix(i,1) << " disparity: " << range_matrix(i,2) << std::endl;
            //angle += angle_increment;
            //std::cout << "i'th: " << i << " range data: " << range_matrix(i,0) << std::endl;
            i++;
        }





        // selecting the furthest distance in the virtual lidar readings (Lidar readings with disparities)
        furthest_distance = range_matrix.col(0).maxCoeff(&furthest_distance_index);
        //std::cout << "furthest distance is: " << furthest_distance << " index is: " << furthest_distance_index << std::endl;
        //std::cout << "car steering angle is: " << starting_angle + furthest_distance_index*angle_increment << std::endl;
        float roll = 0, pitch = 0, yaw = starting_angle + furthest_distance_index*angle_increment; 
        Eigen::Quaternionf q;
        q = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
        //std::cout << "Quaternion: " << q.coeffs() << std::endl;
        //std::cout << q.x() << q.y() << q.z() << q.w() << std::endl;
        direction_arrow.pose.orientation.x = q.x();
        direction_arrow.pose.orientation.y = q.y();
        direction_arrow.pose.orientation.z = q.z();
        direction_arrow.pose.orientation.w = q.w();
        


        // Drive the car based on distance|speed ratio and steering angle
        float steering_angle = starting_angle + furthest_distance_index*angle_increment;
        if(steering_angle > max_steering_angle) {
            steering_angle = max_steering_angle;
        }
        else if(steering_angle < -max_steering_angle) {
            steering_angle = -max_steering_angle;
        }
        else {
            steering_angle = steering_angle;
        }

        if (!std::isfinite(furthest_distance) || furthest_distance > 10.0f) {
           furthest_distance = 10.0f;
        }
        speed = translate(furthest_distance, 0.0, 10.0, min_speed, max_speed); // rad per sec
        if(steering_angle <= 0.1 && steering_angle >= -0.1) {
            speed = speed * 95/100;
        }
        else if(steering_angle <= 0.3 && steering_angle >= -0.3) {
            speed = speed * 65/100;
        }
        else {
            speed = speed * 35/100;
        }
        //speed = std::min(speed, max_speed);
        //std::cout << steering_angle << std::endl;
        //std::cout << speed << std::endl;
        
        drive_cmd.angular.z = steering_angle;
        drive_cmd.linear.x = speed;


        /*
        // publishing is done in odom callback just so it's at the same rate as the sim
        // initialize message to be published
        ackermann_msgs::AckermannDriveStamped drive_st_msg;
        ackermann_msgs::AckermannDrive drive_msg;

        /// SPEED CALCULATION:
        // set constant speed to be half of max speed
        drive_msg.speed = max_speed / 2.0;


        /// STEERING ANGLE CALCULATION
        // random number between 0 and 1
        double random = ((double) rand() / RAND_MAX);
        // good range to cause lots of turning
        double range = max_steering_angle / 2.0;
        // compute random amount to change desired angle by (between -range and range)
        double rand_ang = range * random - range / 2.0;

        // set angle (add random change to previous angle)
        drive_msg.steering_angle = std::min(std::max(prev_angle + rand_ang, -max_steering_angle), max_steering_angle);

        // reset previous desired angle
        prev_angle = drive_msg.steering_angle;

        // set drive message in drive stamped message
        drive_st_msg.drive = drive_msg;

        // publish AckermannDriveStamped message to drive topic
        drive_pub.publish(drive_st_msg);
        */

        auto current_time = msg->header.stamp; // 입력 데이터 시간 복사 (권장)
        // 또는 auto current_time = this->now(); // 현재 노드 시간

        disparity_lidar.header.stamp = current_time;
        direction_arrow.header.stamp = current_time;
        
        // drive_cmd.header.stamp = current_time;
        arrow_marker_pub->publish(direction_arrow);
        drive_pub->publish(drive_cmd);
        lidar_pub->publish(disparity_lidar);



        // Some computation here
        auto end = this->now();
        auto elapsed = (end - start).seconds();
        
        RCLCPP_INFO(this->get_logger(), "Computation finished. Elapsed time: %f s", elapsed);


        // 콜백 함수 마지막 부분에 추가

    }

    // 변수 선언들
    // rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr drive_pub;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr lidar_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr arrow_marker_pub;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub;

    // 파라미터 및 계산용 변수
    double max_speed, min_speed, max_steering_angle;
    float speed, speed_threshold;
    float car_length, car_width, theta, alpha;
    float car_arc_length, car_radius, disp_offset;
    float chord_length, carWidth_tolerance;
    bool after_LD = false;

    // 데이터 저장용
    Eigen::MatrixXd range_matrix = Eigen::MatrixXd::Zero(1081, 3);
    double prev_angle = 0.0;
    
    // 시각화 및 메시지 객체 (ROS 2 타입으로)
    sensor_msgs::msg::LaserScan disparity_lidar;
    visualization_msgs::msg::Marker direction_arrow;
    // ackermann_msgs::msg::AckermannDriveStamped drive_cmd;
    geometry_msgs::msg::Twist drive_cmd;
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