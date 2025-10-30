#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/cvdef.h"
#include <stdio.h>

using namespace cv;

static void help()
{
    printf( "\nOpenCV 卡尔曼滤波器 - 二维平面匀速运动示例\n"
    " 跟踪平面内匀速运动的点。\n"
    " 状态向量: [x, y, vx, vy] - 位置和速度\n"
    " 观测向量: [x, y] - 位置\n"
    " 真实点和测量点用红色线段连接，\n"
    " 真实点和预测点用黄色线段连接，\n"
    " 真实点和修正后的估计点用绿色线段连接。\n"
    " 按任意键（除 ESC 外）将重置跟踪。\n"
    " 按 ESC 键将停止程序。\n"
                );
}

int main(int, char**)
{
    help();
    Mat img(500, 500, CV_8UC3);
    
    // 修改为4状态变量(位置和速度)，2观测变量(位置)
    KalmanFilter KF(4, 2, 0);
    
    Mat state(4, 1, CV_32F); /* (x, y, vx, vy) */
    Mat processNoise(4, 1, CV_32F);
    Mat measurement = Mat::zeros(2, 1, CV_32F);
    char code = (char)-1;

    for(;;)
    {
        img = Scalar::all(0);
        
        // 初始化状态：位置在中心，速度向右下方向
        state.at<float>(0) = img.cols * 0.5f; // x
        state.at<float>(1) = img.rows * 0.5f; // y
        state.at<float>(2) = 2.0f; // vx
        state.at<float>(3) = 1.5f; // vy
        
        // 状态转移矩阵 - 匀速运动模型
        // x_{k+1} = x_k + vx_k * dt
        // y_{k+1} = y_k + vy_k * dt  
        // vx_{k+1} = vx_k
        // vy_{k+1} = vy_k
        KF.transitionMatrix = (Mat_<float>(4, 4) << 
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // 观测矩阵 - 只能观测到位置
        KF.measurementMatrix = (Mat_<float>(2, 4) << 
            1, 0, 0, 0,
            0, 1, 0, 0);

    
        setIdentity(KF.processNoiseCov, Scalar::all(1e-6));    // 过程噪声协方差
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));    // 观测噪声协方差
        setIdentity(KF.errorCovPost, Scalar::all(1));    // 后验误差协方差
        randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));    // 初始化状态后验估计

        for(;;)
        {
            // 获取真实状态点
            Point statePt(state.at<float>(0), state.at<float>(1));

            // 预测步骤
            Mat prediction = KF.predict();
            Point predictPt(prediction.at<float>(0), prediction.at<float>(1));

            // 生成测量值 (加入噪声的真实位置)
            randn(measurement, Scalar::all(0), Scalar::all(sqrt(KF.measurementNoiseCov.at<float>(0, 0))));
            measurement.at<float>(0) += state.at<float>(0); // x测量
            measurement.at<float>(1) += state.at<float>(1); // y测量
            
            Point measPt(measurement.at<float>(0), measurement.at<float>(1));

            // 修正步骤
            KF.correct(measurement);
            Point improvedPt(KF.statePost.at<float>(0), KF.statePost.at<float>(1));

            // 绘制各种点
            img = img * 0.2; // 淡出效果
            
            drawMarker(img, measPt, Scalar(0, 0, 255), cv::MARKER_SQUARE, 5, 2);     // 测量点 - 红色
            drawMarker(img, predictPt, Scalar(0, 255, 255), cv::MARKER_SQUARE, 5, 2);   // 预测点 - 黄色 
            drawMarker(img, improvedPt, Scalar(0, 255, 0), cv::MARKER_SQUARE, 5, 2);    // 修正点 - 绿色
            drawMarker(img, statePt, Scalar(255, 255, 255), cv::MARKER_STAR, 10, 1);    // 真实点 - 白色

            // 绘制连线
            line(img, statePt, measPt, Scalar(0, 0, 255), 1, LINE_AA, 0);    // 红: 真实-测量
            line(img, statePt, predictPt, Scalar(0, 255, 255), 1, LINE_AA, 0); // 黄: 真实-预测
            line(img, statePt, improvedPt, Scalar(0, 255, 0), 1, LINE_AA, 0); // 绿: 真实-修正

            // 更新真实状态 (加入过程噪声)
            randn(processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
            state = KF.transitionMatrix * state + processNoise;

            imshow("Kalman 2D", img);
            code = (char)waitKey(50); // 加快更新速度

            if(code > 0)
                break;
        }
        if(code == 27 || code == 'q' || code == 'Q')
            break;
    }

    return 0;
}


// #include "opencv2/video/tracking.hpp"
// #include "opencv2/highgui.hpp"
// #include "opencv2/core/cvdef.h"
// #include <stdio.h>
 
// using namespace cv;
 
// static inline Point calcPoint(Point2f center, double R, double angle)
// {
//     return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
// }
 
// static void help()
// {
//     printf( "\nOpenCV 卡尔曼滤波器 C 语言调用的示例。\n"
//     " 跟踪旋转点。\n"
//     " 点在一个圆圈内移动，并由一维状态来描述。\n"
//     " state_k+1 = state_k + speed + process_noise N(0, 1e-5)\n"
//     " 速度是恒定的。\n"
//     " 状态和测量向量都是一维的（一个点的角度），\n"
//     " 测量值是真实状态加上高斯噪声 N(0, 1e-1)。\n"
//     " 真实点和测量点用红色线段连接，\n"
//     " 真实点和估计点用黄色线段连接，\n"
//     " 真实点和修正后的估计点用绿色线段连接。\n"
//     " （如果卡尔曼滤波器工作正常，\n"
//     " 黄色线段应该比红色线段短，\n"
//     " 绿色线段应该比黄色线段短）。"
//                 "\n"
//     " 按任意键（除 ESC 外）将重置跟踪。\n"
//     " 按 ESC 键将停止程序。\n"
//                 );
// }
 
// int main(int, char**)
// {
//     help();
//         Mat img(500, 500, CV_8UC3);
//         KalmanFilter KF(2, 1, 0);
//         Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
//         Mat processNoise(2, 1, CV_32F);
//         Mat measurement = Mat::zeros(1, 1, CV_32F);
//         char code = (char)-1;
    
//         for(;;)
//         {
//     img = Scalar::all(0);
//     state.at<float>(0) = 0.0f;
//     state.at<float>(1) = 2.f * (float)CV_PI / 6;
//     KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);
    
//             setIdentity(KF.measurementMatrix);
//             setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
//             setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
//             setIdentity(KF.errorCovPost, Scalar::all(1));
    
//             randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
    
//             for(;;)
//             {
//                 Point2f center(img.cols*0.5f, img.rows*0.5f);
//                 float R = img.cols/3.f;
//                 double stateAngle = state.at<float>(0);
//                 Point statePt = calcPoint(center, R, stateAngle);
    
//                 Mat prediction = KF.predict();
//                 double predictAngle = prediction.at<float>(0);
//                 Point predictPt = calcPoint(center, R, predictAngle);
    
//                 // 生成测量值
//                 randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));
//     measurement += KF.measurementMatrix*state;
    
//                 double measAngle = measurement.at<float>(0);
//                 Point measPt = calcPoint(center, R, measAngle);
    
//                 // 根据测量值修正状态估计值
//                 // 更新 statePost 和 errorCovPost
//     KF.correct(measurement);
//                 double improvedAngle = KF.statePost.at<float>(0);
//                 Point improvedPt = calcPoint(center, R, improvedAngle);
    
//                 // 绘制点
//     img = img * 0.2;
//                 drawMarker(img, measPt, Scalar(0, 0, 255), cv::MARKER_SQUARE, 5, 2);
//                 drawMarker(img, predictPt, Scalar(0, 255, 255), cv::MARKER_SQUARE, 5, 2);

//                 drawMarker(img, improvedPt, Scalar(0, 255, 0), cv::MARKER_SQUARE, 5, 2);

//                 drawMarker(img, statePt, Scalar(255, 255, 255), cv::MARKER_STAR, 10, 1);
//                 // 预测一步
//                 Mat test = Mat(KF.transitionMatrix*KF.statePost);
//                 drawMarker(img, calcPoint(center, R, Mat(KF.transitionMatrix*KF.statePost).at<float>(0)),
//                         Scalar(255, 255, 0), cv::MARKER_SQUARE, 12, 1);
    
//                 line( img, statePt, measPt, Scalar(0,0,255), 1, LINE_AA, 0 );
//                 line( img, statePt, predictPt, Scalar(0,255,255), 1, LINE_AA, 0 );

//                 line( img, statePt, improvedPt, Scalar(0,255,0), 1, LINE_AA, 0 );
    
    
//                 randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
//     state = KF.transitionMatrix*state + processNoise;
    
//                 imshow( "Kalman", img );
//     code = (char)waitKey(1000);
    
//                 if( code > 0 )
//                     break;
//             }
//             if( code == 27 || code == 'q' || code == 'Q' )
//                 break;
//         }
    
//         return 0;
// }