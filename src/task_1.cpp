#include <chrono>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <algorithm>
#include <deque>

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
  int frame_count = 0;
  io::GimbalState gimbal_state;

  const float Max_Yaw = M_PI;
  const float Min_Yaw = -M_PI;
  const float Max_Pitch = M_PI / 9;
  const float Min_Pitch = -M_PI / 9;

  // Yaw轴防抖优化
  float filtered_yaw = 0.0f;
  float last_yaw = 0.0f;
  const float alpha = 0.1f;  // 低通滤波系数
  const float MAX_YAW_DELTA = 0.15f;  // 最大角速度限制
  std::deque<float> yaw_history;  // 滑动窗口滤波
  const int YAW_HISTORY_SIZE = 5;
  const float YAW_DELTA_THRESH = 0.15f;  // 异常跳变阈值

  // 首次识别优化
  bool is_first_detect = true;
  const int TRANSITION_FRAMES = 10;  // 平滑过渡帧数
  int transition_count = 0;
  float start_yaw = 0.0f;
  int first_detect_confirm_count = 0;
  const int CONFIRM_FRAMES = 3;  // 连续确认帧数

  // 视野丢失处理
  int lost_count = 0;
  const int LOST_MAX_COUNT = 10;  // 最大丢失帧数（≈0.33s）
  float last_target_yaw = 0.0f;
  float last_yaw_vel = 0.0f;

  // 运动趋势预判
  std::deque<Eigen::Vector3d> armor_xyz_history;  // 装甲板位置历史
  const int POS_HISTORY_SIZE = 3;  // 位置历史窗口
  Eigen::Vector3d last_armor_xyz;  // 上一帧装甲板位置
  const float YAW_PREDICTION_GAIN = 0.6f;  // 预判增益

  // 云台初始姿态校准
  std::chrono::steady_clock::time_point init_time;
  Eigen::Quaterniond init_q;
  for (int i = 0; i < 3; i++) {
    init_q = gimbal.q(init_time);
    std::this_thread::sleep_for(50ms);
  }
  solver.set_R_gimbal2world(init_q);

  while (!exiter.exit()) {
    camera.read(img, t);
    frame_count++;

    detected_armors = yolo.detect(img, frame_count);
    if (detected_armors.empty()) {
      // 视野丢失处理
      lost_count++;
      if (lost_count <= LOST_MAX_COUNT && last_yaw_vel != 0.0f) {
        float continue_yaw = last_target_yaw + last_yaw_vel * 0.033f;  // 假设30fps
        continue_yaw = std::clamp(continue_yaw, Min_Yaw, Max_Yaw);
        gimbal.send(true, false, continue_yaw, 0.0f);
      } else {
        gimbal.send(false, false, 0.0f, 0.0f);
        yaw_history.clear();
        armor_xyz_history.clear();
        lost_count = 0;
        is_first_detect = true;
        first_detect_confirm_count = 0;
        transition_count = 0;
        last_yaw = 0.0f;
      }
      continue;
    }

    if (is_first_detect) {
      first_detect_confirm_count++;
      if (first_detect_confirm_count < CONFIRM_FRAMES) {
        gimbal.send(false, false, 0.0f, 0.0f);
        continue;
      } else {
        is_first_detect = false;
        gimbal_state = gimbal.state();
        start_yaw = gimbal_state.yaw;
        transition_count = 0;
      }
    }

    gimbal_state = gimbal.state();
    q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    auto_aim::Armor target_armor = detected_armors.front();
    solver.solve(target_armor);
    Eigen::Vector3d armor_world_xyz = target_armor.xyz_in_world;
    Eigen::Matrix3d R_gimbal2world = solver.R_gimbal2world();
    Eigen::Vector3d armor_gimbal_xyz = R_gimbal2world.transpose() * armor_world_xyz;

    float raw_yaw = atan2(armor_gimbal_xyz.y(), armor_gimbal_xyz.x());
    
    float yaw_delta = fabs(raw_yaw - last_target_yaw);
    if (yaw_history.size() > 0 && yaw_delta > YAW_DELTA_THRESH) {
      raw_yaw = last_target_yaw;
    }

    filtered_yaw = alpha * raw_yaw + (1 - alpha) * filtered_yaw;

    yaw_history.push_back(filtered_yaw);
    if (yaw_history.size() > YAW_HISTORY_SIZE) {
      yaw_history.pop_front();
    }
    float avg_yaw = 0.0f;
    for (float y : yaw_history) avg_yaw += y;
    avg_yaw /= yaw_history.size();

    if (armor_gimbal_xyz.norm() > 0.1f) {
      armor_xyz_history.push_back(armor_gimbal_xyz);
      if (armor_xyz_history.size() > POS_HISTORY_SIZE) {
        armor_xyz_history.pop_front();
      }

      Eigen::Vector3d armor_vel(0, 0, 0);
      if (armor_xyz_history.size() >= 2) {
        double time_delta = 0.033;  // 30fps帧间隔
        armor_vel = (armor_gimbal_xyz - last_armor_xyz) / time_delta;
      }
      last_armor_xyz = armor_gimbal_xyz;

      if (fabs(armor_vel.y()) > 0.1f) {
        float predicted_offset = atan2(armor_vel.y(), armor_gimbal_xyz.x()) * YAW_PREDICTION_GAIN;
        avg_yaw += predicted_offset;
      }
    }

    // 角速度限制
    if (last_yaw == 0.0f) last_yaw = avg_yaw;
    float target_yaw = std::clamp(avg_yaw, last_yaw - MAX_YAW_DELTA, last_yaw + MAX_YAW_DELTA);
    last_yaw = target_yaw;

    // 首次识别角度平滑过渡
    if (transition_count < TRANSITION_FRAMES) {
      float transition_ratio = (float)transition_count / TRANSITION_FRAMES;
      target_yaw = start_yaw + (target_yaw - start_yaw) * transition_ratio;
      transition_count++;
    } else {
      target_yaw = std::clamp(target_yaw, Min_Yaw, Max_Yaw);
    }

    float target_pitch = -1.1 * atan2(armor_gimbal_xyz.z(),sqrt(armor_gimbal_xyz.x() * armor_gimbal_xyz.x() + armor_gimbal_xyz.y() * armor_gimbal_xyz.y()));
    target_pitch = std::clamp(target_pitch, Min_Pitch, Max_Pitch);

    gimbal.send(true, false, target_yaw, target_pitch);

    last_target_yaw = target_yaw;
    last_yaw_vel = (target_yaw - last_yaw) / 0.033f;

    double timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t.time_since_epoch()
    ).count();
    nlohmann::json plot_data;
    plot_data["timestamp_ms"] = timestamp_ms;
    plot_data["task"] = "task1";
    plot_data["gimbal_control"]["raw_yaw_rad"] = raw_yaw;
    plot_data["gimbal_control"]["target_yaw_rad"] = target_yaw;
    plot_data["gimbal_control"]["target_pitch_rad"] = target_pitch;
    plot_data["gimbal_state"]["current_yaw_rad"] = gimbal_state.yaw;
    plot_data["gimbal_state"]["current_pitch_rad"] = gimbal_state.pitch;
    plotter.plot(plot_data);

    cv::putText(img, fmt::format("Target Yaw: {:.2f}rad", target_yaw), 
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::putText(img, fmt::format("Target Pitch: {:.2f}rad", target_pitch), 
                cv::Point(20, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
    cv::imshow("task 1", img);
    cv::waitKey(1);
  }

  cv::destroyAllWindows();
  return 0;
}