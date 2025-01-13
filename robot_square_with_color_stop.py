#include "ros/ros.h"
#include <geometry_msgs/Twist.h>
#include "nav_msgs/Odometry.h"
#include <sensor_msgs/Image.h>
#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <string>

// Publicador para la velocidad
ros::Publisher speed_pub;
geometry_msgs::Twist speedMsg;

// Variables para el seguimiento de la posición
double initial_x = 0.0;
double initial_y = 0.0;
bool moving_forward = true;
bool turning = false;
int completed_sides = 0;

// Constantes de movimiento
const double LINEAR_SPEED = 0.2;  // Velocidad lineal (m/s)
const double ANGULAR_SPEED = 0.5; // Velocidad angular (rad/s)
const double FORWARD_DISTANCE = 1.0; // Distancia en metros
const double TURN_ANGLE = M_PI / 2;  // Ángulo en radianes (90 grados)

// Variables para el ángulo
double initial_theta = 0.0;

// Variables para detección de colores
std::string color_detected = "none";

// Función para manejar la detección de colores
void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat hsv;
    cv::cvtColor(cv_ptr->image, hsv, cv::COLOR_BGR2HSV);

    // Rango de color rojo
    cv::Scalar lower_red(0, 100, 100), upper_red(10, 255, 255);
    cv::Mat red_mask;
    cv::inRange(hsv, lower_red, upper_red, red_mask);

    // Rango de color verde
    cv::Scalar lower_green(45, 100, 50), upper_green(75, 255, 255);
    cv::Mat green_mask;
    cv::inRange(hsv, lower_green, upper_green, green_mask);

    if (cv::countNonZero(red_mask) > 0) {
        color_detected = "red";
    } else if (cv::countNonZero(green_mask) > 0) {
        color_detected = "green";
    } else {
        color_detected = "none";
    }

    ROS_INFO("Color detectado: %s", color_detected.c_str());
}

// Función para manejar la odometría
void odometryCallback(const nav_msgs::Odometry::ConstPtr& odomMsg) {
    if (color_detected == "red") {
        // Detenerse si se detecta rojo
        ROS_INFO("Rojo detectado: Deteniéndose.");
        speedMsg.linear.x = 0.0;
        speedMsg.angular.z = 0.0;
        speed_pub.publish(speedMsg);
        return;
    } else if (color_detected == "green") {
        // Avanzar si se detecta verde
        ROS_INFO("Verde detectado: Avanzando.");
        speedMsg.linear.x = LINEAR_SPEED;
        speedMsg.angular.z = 0.0;
        speed_pub.publish(speedMsg);
        return;
    }

    // Si no se detecta color, ejecutar el patrón de cuadrado
    double current_x = odomMsg->pose.pose.position.x;
    double current_y = odomMsg->pose.pose.position.y;

    double siny_cosp = 2 * (odomMsg->pose.pose.orientation.w * odomMsg->pose.pose.orientation.z);
    double cosy_cosp = 1 - 2 * (odomMsg->pose.pose.orientation.z * odomMsg->pose.pose.orientation.z);
    double current_theta = atan2(siny_cosp, cosy_cosp);

    if (moving_forward) {
        double distance = sqrt(pow(current_x - initial_x, 2) + pow(current_y - initial_y, 2));
        if (distance >= FORWARD_DISTANCE) {
            speedMsg.linear.x = 0.0;
            speedMsg.angular.z = 0.0;
            speed_pub.publish(speedMsg);
            moving_forward = false;
            turning = true;
            initial_theta = current_theta;
        } else {
            speedMsg.linear.x = LINEAR_SPEED;
            speedMsg.angular.z = 0.0;
            speed_pub.publish(speedMsg);
        }
    } else if (turning) {
        double angle_turned = fabs(current_theta - initial_theta);
        if (angle_turned >= TURN_ANGLE) {
            speedMsg.linear.x = 0.0;
            speedMsg.angular.z = 0.0;
            speed_pub.publish(speedMsg);
            turning = false;
            moving_forward = true;
            completed_sides++;
            initial_x = current_x;
            initial_y = current_y;

            if (completed_sides >= 4) {
                ros::shutdown();
            }
        } else {
            speedMsg.linear.x = 0.0;
            speedMsg.angular.z = ANGULAR_SPEED;
            speed_pub.publish(speedMsg);
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "square_with_color_detection");
    ros::NodeHandle n;

    // Suscriptores
    ros::Subscriber image_sub = n.subscribe("/camera/rgb/image_raw", 1, imageCallback);
    ros::Subscriber odom_sub = n.subscribe("/odom", 1, odometryCallback);

    // Publicador de velocidades
    speed_pub = n.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    ROS_INFO("Iniciando movimiento con detección de colores...");

    ros::spin();
    return 0;
}
