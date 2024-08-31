#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

std::mt19937 gen(std::random_device{}());
// 定义粒子的结构体，包含状态和权重两个成员变量
struct Particle {
  double state;  // 状态
  double weight; // 权重
};

// 定义状态转移模型的函数，输入为当前状态，输出为预测状态
// 这里只是一个示例，你可以根据你的问题修改这个函数
double state_model(double state) {
  // 定义一个随机数生成器
  // 定义一个正态分布，均值为0，标准差为0.1
  std::normal_distribution<double> normal_dist(0, 0.1);
  // 生成一个随机噪声
  double noise = normal_dist(gen);
  // 返回预测状态，假设状态转移方程为x_t = x_t-1 + noise
  return state + noise;
}

// 定义传感器模型的函数，输入为预测状态和观测值，输出为似然
// 这里只是一个示例，你可以根据你的问题修改这个函数
double sensor_model(double state, double observation) {
  // 定义一个随机数生成器
  // 定义一个正态分布，均值为0，标准差为0.2
  std::normal_distribution<double> normal_dist(0, 0.2);
  // 生成一个随机噪声
  double noise = normal_dist(gen);
  // 返回似然，假设观测方程为y_t = x_t + noise
  return std::exp(-0.5 * std::pow((observation - state - noise) / 0.2, 2)) /
         (0.2 * std::sqrt(2 * M_PI));
}

// 定义粒子滤波器的类，包含粒子集合，状态估计，粒子个数，随机数生成器等成员变量和方法
class ParticleFilter {
private:
  std::vector<Particle> particles;                     // 粒子集合
  double estimate;                                     // 状态估计
  int num_particles;                                   // 粒子个数
  std::uniform_real_distribution<double> uniform_dist; // 均匀分布
  std::discrete_distribution<int> discrete_dist;       // 离散分布

public:
  // 构造函数，初始化粒子滤波器的参数
  ParticleFilter(int n, double min, double max) {
    // 设置粒子个数
    num_particles = n;
    // 设置随机数生成器的种子
    gen.seed(std::random_device()());
    // 设置均匀分布的范围
    uniform_dist = std::uniform_real_distribution<double>(min, max);
    // 初始化粒子集合，根据均匀分布生成初始状态，初始权重设为相等
    for (int i = 0; i < num_particles; i++) {
      Particle p;
      p.state = uniform_dist(gen);
      p.weight = 1.0 / num_particles;
      particles.push_back(p);
    }
    // 初始化状态估计，根据粒子集合计算状态的期望值
    estimate = get_mean();
  }

  // 预测函数，根据状态转移模型对每个粒子进行状态预测
  void predict() {
    for (int i = 0; i < num_particles; i++) {
      // 调用状态转移模型的函数，输入为当前状态，输出为预测状态
      particles[i].state = state_model(particles[i].state);
    }
  }

  // 更新函数，根据传感器模型和观测值对每个粒子进行权重更新
  void update(double observation) {
    // 定义一个临时变量，用于存储权重的和
    double sum = 0;
    for (int i = 0; i < num_particles; i++) {
      // 调用传感器模型的函数，输入为预测状态和观测值，输出为似然
      double likelihood = sensor_model(particles[i].state, observation);
      // 根据似然更新权重
      particles[i].weight = particles[i].weight * likelihood;
      // 累加权重的和
      sum += particles[i].weight;
    }
    // 对权重进行归一化，使其和为1
    for (int i = 0; i < num_particles; i++) {
      particles[i].weight = particles[i].weight / sum;
    }
    // 更新状态估计，根据粒子集合计算状态的期望值
    estimate = get_mean();
  }

  // 重采样函数，根据权重对粒子集合进行重采样
  void resample() {
    // 定义一个新的粒子集合
    std::vector<Particle> new_particles;
    // 设置离散分布的权重，根据粒子的权重
    std::vector<double> weights;

    // 使用 std::transform 函数，把 particles 转换成 weights

    std::transform(particles.begin(), particles.end(),
                   std::back_inserter(weights),
                   [](Particle p) { return p.weight; });

    // 设置离散分布的权重，根据 weights

    discrete_dist =
        std::discrete_distribution<int>(weights.begin(), weights.end());

    // 重复粒子个数次，从原始粒子集合中采样一个粒子，加入新的粒子集合
    for (int i = 0; i < num_particles; i++) {
      int index = discrete_dist(gen);
      new_particles.push_back(particles[index]);
    }
    // 用新的粒子集合替换原始粒子集合
    particles = new_particles;
  }

  // 获取状态估计的函数，返回状态估计的值
  double get_estimate() { return estimate; }

  // 获取状态期望值的函数，根据粒子集合计算状态的期望值
  double get_mean() {
    // 定义一个临时变量，用于存储状态的和
    double sum = 0;
    for (int i = 0; i < num_particles; i++) {
      // 累加状态的和，加权平均
      sum += particles[i].state * particles[i].weight;
    }
    // 返回状态的期望值
    return sum;
  }
};

// 定义一个测试函数，用于模拟观测值和调用粒子滤波器的方法
void test() {
  // 定义一个随机数生成器
  // 定义一个正态分布，均值为0，标准差为0.2
  std::normal_distribution<double> normal_dist(0, 0.2);
  // 定义一个真实状态，初始值为0
  double true_state = 0;
  // 定义一个观测值，初始值为0
  double observation = 0;
  // 定义一个粒子滤波器对象，粒子个数为100，初始状态范围为-1到1
  ParticleFilter pf(100, -1, 1);
  // 重复10次，模拟10个时刻的状态和观测
  for (int t = 0; t < 10; t++) {
    // 根据状态转移模型更新真实状态
    true_state = state_model(true_state);
    // 根据传感器模型生成观测值
    observation = true_state + normal_dist(gen);
    // 调用粒子滤波器的预测函数
    pf.predict();
    // 调用粒子滤波器的更新函数，输入为观测值
    pf.update(observation);
    // 调用粒子滤波器的重采样函数
    pf.resample();
    // 调用粒子滤波器的获取状态估计的函数，输出为状态估计值
    double estimate = pf.get_estimate();
    // 打印真实状态，观测值，和状态估计值
    std::cout << "真实状态: " << true_state << std::endl;
    std::cout << "观测值: " << observation << std::endl;
    std::cout << "状态估计: " << estimate << std::endl;
    std::cout << "------------------------" << std::endl;
  }
}

// 调用测试函数
int main() { test(); }
