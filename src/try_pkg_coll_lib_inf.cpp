//
// Created by olagh on 8/13/23.
//
#include <random>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <collision_lib/collision_lib.hpp>

//#include "visualization_msgs/msg/marker.hpp"
//#include "visualization_msgs/msg/marker_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

struct Particle {
    int id;
    double x;
    double y;
    double z;
    double theta;
};


int main(int argc, char *argv[]) {
    // ros2
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("point_cloud_vis");
    auto pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("point_cloud", 10);

    // Your program's code here
//    std::string directoryPath = "/home/olagh/Desktop/collision_door/";
    std::string directoryPath = "/home/olagh/particle_filter/src/particle_filter/collision_points/collisionpts_hewitthall/";

    int num_particles = 256 * 5;
    std::vector<Particle> particles;

    std::pair<double, double> x_bound = std::make_pair(-5, 5.0);
    std::pair<double, double> y_bound = std::make_pair(-7.0, 7.0);
    std::pair<double, double> z_bound = std::make_pair(-2.0, 2.0);
    std::pair<double, double> theta_bound = std::make_pair(-3.14, 3.14);

    // Add random Gaussian noise to each particle.
    std::default_random_engine gen;
    std::uniform_real_distribution<double> xNoise(x_bound.first, x_bound.second);
    std::uniform_real_distribution<double> yNoise(y_bound.first, y_bound.second);
    std::uniform_real_distribution<double> zNoise(z_bound.first, z_bound.second);
    std::uniform_real_distribution<double> yawNoise(theta_bound.first, theta_bound.second);

    for (int i = 0; i < num_particles; ++i) {
        Particle p = {i, xNoise(gen), yNoise(gen), zNoise(gen), yawNoise(gen)};
        particles.push_back(p);
    }
    std::vector<float> features_inf(3 * num_particles);
    std::vector<float> targets_inf(2 * num_particles);

    for (int i = 0; i < num_particles; ++i) {
        auto &datapoint = particles[i];
        // particles are double while nn takes float
        features_inf[3 * i + 0] = static_cast<float>(datapoint.x);
        features_inf[3 * i + 1] = static_cast<float>(datapoint.y);
        features_inf[3 * i + 2] = static_cast<float>(datapoint.z);
    }

    std::string path = "/home/olagh/particle_filter/src/neural_collision/collision_lib/config/output.json";
    std::string path_config = "/home/olagh/particle_filter/src/neural_collision/collision_lib/config/config.json";

    for (int i = 0; i < 3; i++) {
        nlohmann::json jsonArray;
        std::ifstream inputFile(path);
        if (inputFile.is_open()) {
            inputFile >> jsonArray;
            inputFile.close();
        } else {
            std::cerr << "Unable to open output.json for reading" << std::endl;
        }
        std::vector<float> floatVector = jsonArray.get<std::vector<float>>();
        for (int i = 0; i < 20; i++) {
            check_collision_inf(features_inf, targets_inf, 6, 3,path_config ,floatVector);
            std::cout << "pub" << std::endl;
        }
    }
    return 0;  // Return 0 to indicate successful execution
}