#include "io/my_camera.hpp"
#include "tasks/yolo.hpp"
#include "tasks/armor.hpp"
#include "tools/img_tools.hpp"
#include <opencv2/opencv.hpp>

int main()
{
    // 初始化相机、yolo类
    myCamera camera;
    auto_aim::YOLO yolo("./configs/yolo.yaml");
    cv::Mat img;
    std::vector<auto_aim::Armor> armors;
    int frame_count=0;

    while(1){
        // 调用相机读取图像
        camera.read(img);


        // 调用yolo识别装甲板
        yolo.detect(img,frame_count);

        for(auto &armor:armors)
        {
            tools::draw_points(img,armor.points,cv::Scalar(0,0,255),2);
        }

        // 显示图像
        cv::resize(img, img , cv::Size(640, 480));
        cv::imshow("img", img);
        if (cv::waitKey(0) == 'q') {
            break;
        }

        frame_count++;
    }
    cv::destroyAllWindows();
    return 0;
}