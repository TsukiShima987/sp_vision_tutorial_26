#include <chrono>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <algorithm>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/exiter.hpp"
#include "tools/trajectory.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 初始化工具类
  tools::Exiter exiter;
  tools::Plotter plotter;

  // 初始化io类
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  // 初始化auto_aim类
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Target target;
  bool is_target_init=0;

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  std::list<auto_aim::Armor> detected_armors;
  int frame_count = 0;
  io::GimbalState gimbal_state;

  const float Max_Yaw=M_PI;
  const float Min_Yaw=-M_PI;
  const float Max_Pitch=M_PI/9;
  const float Min_Pitch=-M_PI/9;
  const double GRAVITY=9.7833;

  // EKF初始化参数
  Eigen::VectorXd P0_dig;//=Eigen::VectorXd::Ones(11)*0.01;
  P0_dig << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.01, 0.01, 0.01;

  int current_shots = 0;

  while (!exiter.exit()) {

    camera.read(img, t);
    frame_count++;

    detected_armors = yolo.detect(img, frame_count);
    if(detected_armors.empty()){
      continue;
    }

    gimbal_state = gimbal.state();
    q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    auto_aim::Armor target_armor=detected_armors.front();
    solver.solve(target_armor);

    if (!is_target_init) {
      target=auto_aim::Target(target_armor, t, P0_dig, 0.2, 4);
      is_target_init=1;
    } else {
      target.predict(t);
      target.update(target_armor);
    }
    

    double distance = target_armor.ypd_in_world[2];
    double bullet_speed = gimbal_state.bullet_speed;

    float target_pitch=-tools::Trajectory(bullet_speed,distance,target_armor.xyz_in_world[2]).pitch;
    
    float target_yaw=target_armor.ypd_in_world[0];
    // if (target.convergened()) {
    //   target_yaw = target.ekf_x()[6];
    // }

    // target_yaw = std::clamp(target_yaw, Min_Yaw, Max_Yaw);
    // target_pitch = std::clamp(target_pitch, Min_Pitch, Max_Pitch);

    if(target.convergened()){
      gimbal.send(1,1,target_yaw, target_pitch);
      current_shots++;
    }

    double timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t.time_since_epoch()
    ).count();
    nlohmann::json plot_data;
    plot_data["timestamp_ms"] = timestamp_ms;
    plot_data["task"] = "task3";
    plot_data["gimbal_control"]["target_yaw_rad"] = target_yaw;
    plot_data["gimbal_control"]["target_pitch_rad"] = target_pitch;
    plot_data["gimbal_state"]["current_yaw_rad"] = gimbal_state.yaw;
    plot_data["gimbal_state"]["current_pitch_rad"] = gimbal_state.pitch;
    
    if (is_target_init) {
      plot_data["ekf"]["angular_velocity_rad/s"] = target.ekf_x()[7];
      plot_data["ekf"]["is_converged"] = target.convergened() ? 1 : 0;
    }
    plotter.plot(plot_data);

    cv::putText(img, fmt::format("Distance：{:.2f}m | Shot Speed：{:.2f}m/s",
      distance, bullet_speed),
      cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2);
    if (is_target_init) {
      cv::putText(img, fmt::format("EKF w：{:.2f}rad/s | Converged：{}",
        target.ekf_x()[7], target.convergened() ? "1" : "0"),
        cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,255), 2);
    }

    cv::imshow("task3", img);
    cv::waitKey(1);
  }

  cv::destroyAllWindows();
  return 0;
}