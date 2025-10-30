#include "ArmorSim.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

int main() {
    // 创建模拟器
    ArmorPlateSimulator simulator(0.01, 0.5, 0.1);  // Δt=10ms, 观测噪声σ=0.5, 过程噪声σ=0.1

    double simulation_time = 1;  // 模拟1秒

    std::cout << "装甲板运动模拟 (帧率: 100fps, Δt = 10ms)" << std::endl;
    std::cout << "初始位置: (0, 0), 初始速度: (2, 3)" << std::endl;
    std::cout << "观测噪声标准差: 0.5, 过程噪声标准差: 0.1" << std::endl;

    // 恒定速度运动模拟
    auto constant_velocity = simulator.simulateConstantVelocity(simulation_time);
    simulator.printResults(constant_velocity, "恒定速度运动");

    // 带观测噪声的模拟
    auto with_measurement_noise = simulator.simulateWithMeasurementNoise(simulation_time);
    simulator.printNoisyResults(with_measurement_noise, "带观测噪声的运动");

    // 带过程噪声和观测噪声的模拟
    auto with_both_noises = simulator.simulateWithBothNoises(simulation_time);
    simulator.printNoisyResults(with_both_noises, "带过程噪声和观测噪声的运动");

    return 0;
}
