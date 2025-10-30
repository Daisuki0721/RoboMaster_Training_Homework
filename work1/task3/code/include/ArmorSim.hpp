#ifndef _ARMORSIM_HPP_
#define _ARMORSIM_HPP_

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

// 运动状态结构体
struct MotionState {
    std::vector<double> position;
    std::vector<double> velocity;
    MotionState() : position(2, 0.0), velocity(2, 0.0) {}
    MotionState(std::vector<double> pos, std::vector<double> vel) 
        : position(pos), velocity(vel) {}
};

class ArmorPlateSimulator {
private:
    double delta_t;
    double measurement_noise_std;  // 观测噪声标准差
    double process_noise_std;      // 过程噪声标准差

    // 随机数生成器
    std::default_random_engine generator;
    std::normal_distribution<double> measurement_noise_dist;
    std::normal_distribution<double> process_noise_dist;

    MotionState initial_state;

public:
    ArmorPlateSimulator(double dt = 0.01, double meas_std = 0.5, double proc_std = 0.1);           // 构造函数

    std::vector<MotionState> simulateConstantVelocity(double total_time);       // 恒定速度无噪声运动模型

    std::vector<MotionState> simulateWithMeasurementNoise(double total_time);   // 生成带观测噪声的观测位置

    std::vector<MotionState> simulateWithBothNoises(double total_time);         // 引入过程噪声的运动模型

    void printResults(const std::vector<MotionState>& states, const std::string& title);    // 打印结果

    void printNoisyResults(const std::vector<MotionState>& results, const std::string& title);
};

#endif
