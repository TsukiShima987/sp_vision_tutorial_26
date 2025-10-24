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

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  std::list<auto_aim::Armor> detected_armors;
  int frame_count=0;
  io::GimbalState gimbal_state;

  const float Max_Yaw=M_PI;
  const float Min_Yaw=-M_PI;
  const float Max_Pitch=M_PI/9;
  const float Min_Pitch=-M_PI/9;  
  const double GRAVITY = 9.7833; 

  while (!exiter.exit()) {
    // Your code start

    camera.read(img,t);
    frame_count++;

    detected_armors=yolo.detect(img,frame_count);
    if(detected_armors.empty()){
      continue;
    }

    gimbal_state=gimbal.state();
    q=gimbal.q(t);

    solver.set_R_gimbal2world(q);
    auto_aim::Armor target_armor=detected_armors.front();
    solver.solve(target_armor);

    float target_yaw = target_armor.ypd_in_world[0];
    float base_pitch = -target_armor.ypd_in_world[1];

    double distance = target_armor.ypd_in_world[2];
    double bullet_speed = gimbal_state.bullet_speed;
    float target_pitch = -tools::Trajectory(bullet_speed,distance,target_armor.xyz_in_world).pitch;

    gimbal.send(1,1,target_yaw,target_pitch);

    double timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t.time_since_epoch()
    ).count();
    nlohmann::json plot_data;
    plot_data["timestamp_ms"] = timestamp_ms;
    plot_data["task"] = "task2";
    plot_data["gimbal_control"]["target_yaw_rad"] = target_yaw;
    plot_data["gimbal_control"]["target_pitch_rad"] = target_pitch;
    plot_data["gimbal_state"]["current_yaw_rad"] = gimbal_state.yaw;
    plot_data["gimbal_state"]["current_pitch_rad"] = gimbal_state.pitch;
    plotter.plot(plot_data);

    cv::putText(img,fmt::format("Target Yaw: {:.2f}rad", target_yaw), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::putText(img,fmt::format("Target Pitch: {:.2f}rad", target_pitch), cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::imshow("task 2", img);
    cv::waitKey(1);
    // Your code end
  }
  cv::destroyAllWindows();
  return 0;
}