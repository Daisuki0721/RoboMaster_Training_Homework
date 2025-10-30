#include "ArmorSim.hpp"

// 构造函数
ArmorPlateSimulator::ArmorPlateSimulator(double dt, double meas_std, double proc_std)
    : delta_t(dt), 
    measurement_noise_std(meas_std), 
    process_noise_std(proc_std),
    measurement_noise_dist(0.0, meas_std), 
    process_noise_dist(0.0, proc_std),
    initial_state({{0.0f, 0.0f}, {2.0f, 3.0f}}) {   
    }

// 恒定速度无噪声运动模型
std::vector<MotionState> ArmorPlateSimulator::simulateConstantVelocity(double total_time) {
    int steps = static_cast<int>(total_time / delta_t) + 1;
    std::vector<MotionState> states;
    states.resize(steps);

    MotionState current_state = initial_state;

    for (int i = 0; i < steps; i++) {
        states[i] = current_state;

        // 恒定速度运动模型
        current_state.position[0] += current_state.velocity[0] * delta_t;
        current_state.position[1] += current_state.velocity[1] * delta_t;
    }

    return states;
}

// 生成带观测噪声的观测位置
std::vector<MotionState> ArmorPlateSimulator::simulateWithMeasurementNoise(double total_time) {
    auto true_states = simulateConstantVelocity(total_time);
    std::vector<MotionState> results;
    results.resize(true_states.size(), initial_state);

    for (int i = 0; i < true_states.size(); i++) {
        // 添加高斯噪声到观测位置
        results[i].position[0] = true_states[i].position[0] + measurement_noise_dist(generator);
        results[i].position[1] = true_states[i].position[1] + measurement_noise_dist(generator);
    }

    return results;
}

// 引入过程噪声的运动模型
std::vector<MotionState> ArmorPlateSimulator::simulateWithBothNoises(double total_time) {
    int steps = static_cast<int>(total_time / delta_t) + 1;
    std::vector<MotionState> results;
    results.resize(steps);

    MotionState current_state = initial_state;

    for (int i = 0; i < steps; i++) {
        // 添加观测噪声
        results[i].position[0] = current_state.position[0] + measurement_noise_dist(generator);
        results[i].position[1] = current_state.position[1] + measurement_noise_dist(generator);

        // 更新运动速度
        results[i].velocity[0] = current_state.velocity[0];
        results[i].velocity[1] = current_state.velocity[1];

        // 添加过程噪声
        current_state.velocity[0] += process_noise_dist(generator);
        current_state.velocity[1] += process_noise_dist(generator);

        // 更新位置
        current_state.position[0] += results[i].velocity[0] * delta_t;
        current_state.position[1] += results[i].velocity[1] * delta_t;
    }

    return results;
}

// 打印结果
void ArmorPlateSimulator::printResults(const std::vector<MotionState>& states, const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::setw(6) << "Time" << std::setw(10) << "True X"
        << std::setw(10) << "True Y" << std::setw(10) << "Vel X"
        << std::setw(10) << "Vel Y" << std::endl;
    std::cout << std::string(46, '-') << std::endl;

    for (size_t i = 0; i < states.size(); ++i) {
        double time = i * delta_t;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(6) << time
            << std::setw(10) << states[i].position[0]
            << std::setw(10) << states[i].position[1]
            << std::setw(10) << states[i].velocity[0]
            << std::setw(10) << states[i].velocity[1] << std::endl;
    }
}

void ArmorPlateSimulator::printNoisyResults(const std::vector<MotionState>& results,
    const std::string& title) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::setw(6) << "Time" << std::setw(12) << "Observed X"
        << std::setw(12) << "Observed Y" << std::setw(12) << "Noisy VX"
        << std::setw(12) << "Noisy VY" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (size_t i = 0; i < results.size(); ++i) {
        double time = i * delta_t;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(6) << time
            << std::setw(12) << results[i].position[0]
            << std::setw(12) << results[i].position[1]
            << std::setw(12) << results[i].velocity[0]
            << std::setw(12) << results[i].velocity[1] << std::endl;
    }
}