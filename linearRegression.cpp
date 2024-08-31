#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// 计算x的平均值

double mean(const vector<double> &x) {
  double sum = 0;
  for (double xi : x) {
    sum += xi;
  }
  return sum / x.size();
}

// 计算x和y的点积

double dot(const vector<double> &x, const vector<double> &y) {
  double sum = 0;
  for (int i = 0; i < x.size(); i++) {
    sum += x[i] * y[i];
  }
  return sum;
}

// 计算x的平方和

double square_sum(const vector<double> &x) {
  double sum = 0;
  for (double xi : x) {
    sum += xi * xi;
  }
  return sum;
}

// 计算线性回归的参数a和b

void linear_regression(const vector<double> &x, const vector<double> &y,
                       double &a, double &b) {
  // 检查x和y的长度是否相等
  if (x.size() != y.size()) {
    cout << "Error: x and y must have the same size." << endl;
    return;
  }

  // 检查x和y是否为空
  if (x.empty() || y.empty()) {
    cout << "Error: x and y must not be empty." << endl;
    return;
  }

  // 计算x和y的平均值
  double x_mean = mean(x);
  double y_mean = mean(y);

  // 计算a和b的分子和分母
  double numerator_a = y_mean * square_sum(x) - x_mean * dot(x, y);
  double numerator_b = dot(x, y) - x.size() * x_mean * y_mean;
  double denominator = square_sum(x) - x.size() * x_mean * x_mean;

  // 检查分母是否为零
  if (denominator == 0) {
    cout << "Error: x has zero variance." << endl;
    return;
  }

  // 计算a和b
  a = numerator_a / denominator;
  b = numerator_b / denominator;
}

// 测试代码

int main() {

  // 创建一些测试数据
  vector<double> x = {1, 2, 3, 4, 5};
  vector<double> y = {2, 3, 5, 6, 8};
  // 计算线性回归的参数
  double a, b;
  linear_regression(x, y, a, b);

  // 输出结果
  cout << "The linear regression equation is y = " << a << " + " << b << "x"
       << endl;
  return 0;
}
