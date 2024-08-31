// 引入必要的头文件
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

// 定义一个函数，用来打印一个向量
void print_vector(vector<double> u) {
  cout << "[";
  for (int i = 0; i < u.size(); i++) {
    cout << u[i];
    if (i < u.size() - 1) {
      cout << ", ";
    }
  }
  cout << "]";
}

// 定义一个函数，用来打印一个矩阵
void print_matrix(vector<vector<double>> A) {
  cout << "[ ";
  for (int i = 0; i < A.size(); i++) {
    if (i != 0)
      cout << "  ";
    print_vector(A[i]);
    if (i < A.size() - 1) {
      cout << ", \n";
    }
  }
  cout << " ]" << endl;
}
// 定义一个函数，用来计算两个向量的点积
double dot_product(vector<double> u, vector<double> v) {
  double result = 0;
  for (int i = 0; i < u.size(); i++) {
    result += u[i] * v[i];
  }
  return result;
}

// 定义一个函数，用来计算一个向量的长度
double norm(vector<double> u) { return sqrt(dot_product(u, u)); }

// 定义一个函数，用来计算一个向量的正交投影
vector<double> projection(vector<double> u, vector<double> v) {
  double scale = dot_product(u, v) / dot_product(v, v);
  vector<double> result;
  for (int i = 0; i < v.size(); i++) {
    result.push_back(scale * v[i]);
  }
  return result;
}

// 定义一个函数，用来计算一个矩阵的转置
vector<vector<double>> transpose(vector<vector<double>> A) {
  vector<vector<double>> result;
  for (int i = 0; i < A[0].size(); i++) {
    vector<double> row;
    for (int j = 0; j < A.size(); j++) {
      row.push_back(A[j][i]);
    }
    result.push_back(row);
  }
  return result;
}

// 定义一个函数，用来计算两个矩阵的乘积
vector<vector<double>> matrix_product(vector<vector<double>> A,
                                      vector<vector<double>> B) {
  vector<vector<double>> result;
  for (int i = 0; i < A.size(); i++) {
    vector<double> row;
    for (int j = 0; j < B[0].size(); j++) {
      double sum = 0;
      for (int k = 0; k < A[0].size(); k++) {
        sum += A[i][k] * B[k][j];
      }
      row.push_back(sum);
    }
    result.push_back(row);
  }
  return result;
}

// 定义一个函数，用来计算一个矩阵的逆
vector<vector<double>> inverse(vector<vector<double>> A) {
  // 假设A是一个方阵，并且可逆
  int n = A.size();
  // 创建一个单位矩阵
  vector<vector<double>> I;
  for (int i = 0; i < n; i++) {
    vector<double> row;
    for (int j = 0; j < n; j++) {
      if (i == j) {
        row.push_back(1);
      } else {
        row.push_back(0);
      }
    }
    I.push_back(row);
  }
  // 用高斯消元法将A变成上三角矩阵，同时对I做相同的操作
  for (int i = 0; i < n; i++) {
    // 找到第i列中第i行以下的最大的元素
    double max = abs(A[i][i]);
    int max_row = i;
    for (int j = i + 1; j < n; j++) {
      if (abs(A[j][i]) > max) {
        max = abs(A[j][i]);
        max_row = j;
      }
    }
    // 如果最大的元素是0，那么矩阵不可逆，返回空矩阵
    if (max == 0) {
      return vector<vector<double>>();
    }
    // 如果最大的元素不在第i行，那么交换第i行和最大元素所在的行
    if (max_row != i) {
      swap(A[i], A[max_row]);
      swap(I[i], I[max_row]);
    }
    // 将第i行的第i个元素变成1
    double temp = A[i][i];
    for (int j = i; j < n; j++) {
      A[i][j] /= temp;
    }
    for (int j = 0; j < n; j++) {
      I[i][j] /= temp;
    }
    // 将第i列的其他元素变成0
    for (int j = 0; j < n; j++) {
      if (j != i) {
        temp = A[j][i];
        for (int k = i; k < n; k++) {
          A[j][k] -= temp * A[i][k];
        }
        for (int k = 0; k < n; k++) {
          I[j][k] -= temp * I[i][k];
        }
      }
    }
  }
  // 此时，A变成了单位矩阵，而I变成了A的逆矩阵，返回I
  return I;
}

// 定义一个函数，用来计算线性回归的最佳参数
vector<double> linear_regression(vector<vector<double>> X, vector<double> y) {
  // 假设X是一个n x m的矩阵，y是一个n维的向量，我们要求一个m维的向量w，使得y -
  // Xw的长度最小 根据正交投影的原理，我们有(X^T X)w = X^T y，其中X^T是X的转置
  // 如果X^T X是可逆的，那么我们可以求出w = (X^T X)^(-1) X^T y，其中(X^T
  // X)^(-1)是X^T X的逆矩阵 我们用上面定义的函数来计算w
  vector<vector<double>> XT = transpose(X);           // 计算X的转置
  vector<vector<double>> XTX = matrix_product(XT, X); // 计算X^T X
  vector<vector<double>> XTXI = inverse(XTX);         // 计算X^T X的逆
  vector<vector<double>> XTY =
      matrix_product(XT, transpose(vector<vector<double>>(
                             {y}))); // 计算X^T y，注意要把y转成一个n x 1的矩阵
  vector<vector<double>> W =
      matrix_product(XTXI, XTY); // 计算w，注意w是一个m x 1的矩阵
  // 我们将w转成一个m维的向量，并返回
  vector<double> result;
  for (int i = 0; i < W.size(); i++) {
    result.push_back(W[i][0]);
  }
  return result;
}

// 定义一个主函数，用来测试线性回归的函数
int main() {
  // 创建一个X矩阵，表示自变量
  vector<vector<double>> X = {{1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}};
  // 创建一个y向量，表示因变量
  vector<double> y = {2, 4, 6, 8, 10};
  // 调用线性回归的函数，得到最佳的参数w
  vector<double> w = linear_regression(X, y);
  // 打印结果
  cout << "The best parameters are: " << endl;
  print_vector(w);
  return 0;
}
