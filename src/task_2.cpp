#include <chrono>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <list>

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

  const int TARGET_TOTAL=3;
  const int MAX_SHOTS_PER_TARGET=10;
  const double MAX_DISTANCE=5.0;
  const double GRAVITY=9.8;

  int current_target=0;
  int current_shots=0;
  bool is_target_completed=false;
  bool is_waiting_target=true;

  const float Max_Yaw=M_PI;
  const float Min_Yaw=-M_PI;
  const float Max_Pitch=M_PI/9;
  const float Min_Pitch=-M_PI/9;

  while (!exiter.exit()) {
    // Your code start
    camera.read(img,t);
    frame_count++;

    detected_armors=yolo.detect(img,frame_count);
    if(is_target_completed){
      cv::imshow("task 2",img);
      if(cv::waitKey(1)>=0){
        current_target++;
        current_shots=0;
        is_target_completed=false;
        is_waiting_target=false;
      }
      continue;
    }
    std::list<auto_aim::Armor> valid_armors;
    for(const auto& armor : detected_armors){
      double armor_distance=armor.xyz_in_gimbal.norm();
      if(armor_distance<=MAX_DISTANCE&&!armor.duplicated){
        valid_armors.emplace_back(armor);
      }
    }
    if(valid_armors.empty()){
      gimbal.send(0,0,0.0f,0.0f);
      cv::imshow("task 2", img);
      cv::waitKey(1);
      continue;
    }
    auto target_armor=*std::min_element(valid_armors.begin(),valid_armors.end(),[](const auto& a, const auto& b){
      return a.xyz_in_gimbal.norm()<b.xyz_in_gimbal.norm();
    });

    gimbal_state=gimbal.state();
    q=gimbal.q(t);

    solver.set_R_gimbal2world(q);
    solver.solve(target_armor);

    Eigen::Vector3d armor_world_xyz=target_armor.xyz_in_world;
    Eigen::Matrix3d R_gimbal2world=solver.R_gimbal2world();
    Eigen::Vector3d armor_gimbal_xyz=R_gimbal2world.transpose()*armor_world_xyz;

    double target_distance=armor_gimbal_xyz.norm();
    double projectile_speed=gimbal_state.bullet_speed;
    double flight_time=target_distance/projectile_speed;
    double drop_distance=0.5*GRAVITY*flight_time*flight_time;

    float target_yaw=atan2(armor_gimbal_xyz.y(),armor_gimbal_xyz.x());
    float base_pitch=atan2(armor_gimbal_xyz.z(),sqrt(armor_gimbal_xyz.x()*armor_gimbal_xyz.x()+armor_gimbal_xyz.y()*armor_gimbal_xyz.y()));
    float pitch_compensate=atan2(drop_distance,target_distance);
    float target_pitch=base_pitch+pitch_compensate;

    target_yaw=std::clamp(target_yaw,Min_Yaw,Max_Yaw);
    target_pitch=std::clamp(target_pitch,Min_Pitch,Max_Pitch);

    if(current_shots<MAX_SHOTS_PER_TARGET&&!is_waiting_target){
      gimbal.send(1,0,target_yaw,target_pitch);
      std::this_thread::sleep_for(200ms);

      gimbal.send(1,1,target_yaw,target_pitch);
      current_shots++;
      
      std::this_thread::sleep_for(100ms);
      gimbal.send(1,0,target_yaw,target_pitch);

      if(current_shots>=MAX_SHOTS_PER_TARGET){
        is_target_completed=true;
        gimbal.send(0,0,0.0f,0.0f);
      }
    } 
    else if(current_shots>=MAX_SHOTS_PER_TARGET&&!is_target_completed) {
      is_target_completed=true;
      gimbal.send(0,0,0.0f,0.0f);
    }

    double timestamp_ms=std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
    nlohmann::json plot_data;
    plot_data["timestamp_ms"]=timestamp_ms;
    plot_data["task"]="task2";
    plot_data["gimbal_control"]["target_yaw_rad"]=target_yaw;
    plot_data["gimbal_control"]["target_pitch_rad"]=target_pitch;
    plot_data["gimbal_control"]["is_fire"]=(current_shots<MAX_SHOTS_PER_TARGET)?1:0;
    plot_data["gimbal_state"]["current_yaw_rad"]=gimbal_state.yaw;
    plot_data["gimbal_state"]["current_pitch_rad"]=gimbal_state.pitch;
    plot_data["target_info"]["distance_m"]=target_distance;
    plot_data["target_info"]["shots_used"]=current_shots;
    plot_data["target_info"]["drop_distance_m"]=drop_distance;
    plotter.plot(plot_data);

    cv::putText(img,fmt::format("task2：Target{}（{}/{}）", 
      current_target + 1, current_shots, MAX_SHOTS_PER_TARGET), 
      cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
    cv::putText(img, fmt::format("Target Distance：{:.2f}m | Shot Speed：{:.2f}m/s", 
      target_distance, projectile_speed), 
      cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
    cv::putText(img, fmt::format("Target Yaw：{:.2f}rad | Target Pitch：{:.2f}rad", 
      target_yaw, target_pitch), 
      cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::imshow("task 2", img);
    cv::waitKey(1);
    // Your code end
  }
  cv::destroyAllWindows();
  return 0;
}