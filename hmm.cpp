// 导入标准库，用于输入输出和向量运算
#include <iostream>
#include <vector>
using namespace std;

template <class T> ostream &operator<<(ostream &ostr, std::vector<T> u) {
  ostr << "[";
  for (int i = 0; i < u.size(); i++) {
    ostr << u[i];
    if (i < u.size() - 1) {
      ostr << ", ";
    }
  }
  ostr << "]";
  return ostr;
}
template <class T>
ostream &operator<<(ostream &ostr, std::vector<vector<T>> u) {
  ostr << "[ ";
  for (int i = 0; i < u.size(); i++) {
    ostr << u[i];
    if (i < u.size() - 1) {
      ostr << ", \n  ";
    }
  }
  ostr << " ]";
  return ostr;
}
// 定义hmm模型的参数，你可以根据你的问题修改这些参数
// 状态转移概率矩阵A，大小为n*n，n为隐藏状态的个数
std::vector<std::vector<double>> A = {
    {0.5, 0.2, 0.3}, {0.3, 0.5, 0.2}, {0.2, 0.3, 0.5}};
// 观测概率矩阵B，大小为n*m，m为观测值的个数
std::vector<std::vector<double>> B = {
    {0.5, 0.4, 0.1}, {0.1, 0.3, 0.6}, {0.3, 0.3, 0.4}};
// 初始状态概率向量pi，大小为n
std::vector<double> pi = {0.2, 0.4, 0.4};

// 定义观测序列f_i，你可以根据你的数据修改这个序列
std::vector<int> f_i = {2, 1, 2, 0, 1};

// 定义viterbi算法的函数，输入为观测序列f_i，输出为最优隐藏状态序列w_i
std::vector<int> viterbi(std::vector<int> f_i) {
  // 获取观测序列的长度T
  int T = f_i.size();
  // 获取隐藏状态的个数n
  int n = sizeof(pi) / sizeof(pi[0]);
  // 定义动态规划的表格delta，大小为n*T，用于存储每个时刻每个状态的最大概率
  vector<vector<double>> delta(n, vector<double>(T, 0));
  // 定义回溯的表格psi，大小为n*T，用于存储每个时刻每个状态的最优前驱状态
  vector<vector<int>> psi(n, vector<int>(T, 0));
  // 定义最优隐藏状态序列w_i，大小为T
  std::vector<int> w_i(T);

  // 初始化delta和psi的第一列，根据初始状态概率向量pi和观测概率矩阵B
  for (int i = 0; i < n; i++) {
    delta[i][0] = pi[i] * B[i][f_i[0]];
  }
  // 递推计算delta和psi的剩余列，根据状态转移概率矩阵A和观测概率矩阵B
  for (int t = 1; t < T; t++) {
    for (int i = 0; i < n; i++) {
      // 定义一个临时变量max_delta，用于存储当前时刻当前状态的最大概率
      double max_delta = 0;
      // 定义一个临时变量max_psi，用于存储当前时刻当前状态的最优前驱状态
      int max_psi = 0;
      // 遍历所有可能的前驱状态，找出最大概率和最优前驱状态
      for (int j = 0; j < n; j++) {
        // 计算前驱状态j到当前状态i的概率
        double delta_ji = delta[j][t - 1] * A[j][i] * B[i][f_i[t]];
        // cout << endl
        //      << delta[j][t - 1] << " * " << A[j][i] << " * " << B[i][f_i[t]]
        //      << " = " << delta_ji;
        // 如果概率大于当前的最大概率，更新最大概率和最优前驱状态
        if (delta_ji > max_delta) {
          cout << "!";
          max_delta = delta_ji;
          max_psi = j;
        }
      }
      // 将最大概率和最优前驱状态存入delta和psi的表格中
      delta[i][t] = max_delta;
      psi[i][t] = max_psi;
    }
  }

  // 回溯找出最优隐藏状态序列w_i，从最后一个时刻开始
  // 定义一个临时变量max_delta，用于存储最后一个时刻的最大概率
  double max_delta = 0;
  // 定义一个临时变量max_i，用于存储最后一个时刻的最优状态
  int max_i = 0;
  // 遍历所有可能的状态，找出最大概率和最优状态
  for (int i = 0; i < n; i++) {
    // 如果概率大于当前的最大概率，更新最大概率和最优状态
    if (delta[i][T - 1] > max_delta) {
      max_delta = delta[i][T - 1];
      max_i = i;
    }
  }
  // 将最优状态存入w_i的最后一个元素中
  w_i[T - 1] = max_i;

  // 从最后一个时刻向前回溯，根据psi的表格找出每个时刻的最优状态，存入w_i中
  for (int t = T - 2; t >= 0; t--) {
    w_i[t] = psi[w_i[t + 1]][t + 1];
  }

  cout << "\n动态规划的表格delta\n"
       << delta << endl
       << "回溯的表格psi\n"
       << psi << endl;
  // 返回最优隐藏状态序列w_i
  return w_i;
}

int main() {
  // 调用viterbi算法的函数，输入为观测序列f_i，输出为最优隐藏状态序列w_i
  std::vector<int> w_i = viterbi(f_i);

  std::cout << "状态转移概率矩阵A:\n" << A << std::endl;
  std::cout << "观测概率矩阵B:\n" << B << std::endl;
  std::cout << "初始状态概率pi:\n" << pi << std::endl;
  std::cout << "观测序列f_i: \n" << f_i << std::endl;
  std::cout << "最优隐藏状态序列w_i: " << w_i << std::endl;
}
