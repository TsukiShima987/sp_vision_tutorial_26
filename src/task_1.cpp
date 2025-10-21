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

  float filtered_yaw=0.0f;
  float last_yaw=0.0f;
  const float alpha=0.3f;
  const float MAX_YAW_DELTA=0.1f;

  while (!exiter.exit()) {
    // Your code start
    camera.read(img,t);
    frame_count++;

    detected_armors=yolo.detect(img,frame_count);
    if(detected_armors.empty()){
      gimbal.send(0,0,0.0f,0.0f);
      continue;
    }

    gimbal_state=gimbal.state();
    q=gimbal.q(t);

    solver.set_R_gimbal2world(q);
    auto_aim::Armor target_armor=detected_armors.front();
    solver.solve(target_armor);

    Eigen::Vector3d armor_world_xyz=target_armor.xyz_in_world;
    Eigen::Matrix3d R_gimbal2world=solver.R_gimbal2world(); 
    Eigen::Vector3d armor_gimbal_xyz=R_gimbal2world.transpose()*armor_world_xyz;

    float raw_yaw=atan2(armor_gimbal_xyz.y(),armor_gimbal_xyz.x());
    filtered_yaw=alpha*raw_yaw+(1-alpha)*filtered_yaw;
    float target_yaw=std::clamp(filtered_yaw,last_yaw-MAX_YAW_DELTA,last_yaw+MAX_YAW_DELTA);
    last_yaw=target_yaw;
    float target_pitch=-1.1*atan2(armor_gimbal_xyz.z(),sqrt(armor_gimbal_xyz.x()*armor_gimbal_xyz.x()+armor_gimbal_xyz.y()*armor_gimbal_xyz.y()));

    target_yaw=std::clamp(target_yaw,Min_Yaw,Max_Yaw);
    target_pitch=std::clamp(target_pitch,Min_Pitch,Max_Pitch);

    gimbal.send(1,0,target_yaw,target_pitch);

    double timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t.time_since_epoch()
    ).count();
    nlohmann::json plot_data;
    plot_data["timestamp_ms"] = timestamp_ms;
    plot_data["task"] = "task1";
    plot_data["gimbal_control"]["target_yaw_rad"] = target_yaw;
    plot_data["gimbal_control"]["target_pitch_rad"] = target_pitch;
    plot_data["gimbal_state"]["current_yaw_rad"] = gimbal_state.yaw;
    plot_data["gimbal_state"]["current_pitch_rad"] = gimbal_state.pitch;
    plotter.plot(plot_data);

    cv::putText(img,fmt::format("Target Yaw: {:.2f}rad", target_yaw), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::putText(img,fmt::format("Target Pitch: {:.2f}rad", target_pitch), cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::imshow("task 1", img);
    cv::waitKey(1);
    // Your code end
  }
  cv::destroyAllWindows();
  return 0;
}