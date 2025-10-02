#ifndef MY_CAMERA_HPP
#define MY_CAMERA_HPP

#include "hikrobot/include/MvCameraControl.h"
#include <opencv2/opencv.hpp>
#include <unordered_map>

cv::Mat transfer(MV_FRAME_OUT& raw);

class myCamera{
    public:
        myCamera();
        void read(cv::Mat &output_mat);
        ~myCamera();
    private:
        void * handle_;
        int ret_;
        MV_CC_DEVICE_INFO_LIST device_list_;
        MV_FRAME_OUT raw_;

        cv::Mat transfer(MV_FRAME_OUT& raw);
};

#endif