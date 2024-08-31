#include "print.cpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

using MT = vector<vector<double>>;
using VT = vector<double>;
using Eigen::EigenSolver;
using Eigen::Map;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::Success;

MT matmul(MT x, MT y) {
  assert(x.size() == y[0].size());

  MT ans;
  for (int i = 0; i < x.size(); i++) {
    VT row;
    for (int j = 0; j < y[0].size(); j++) {
      double sum = 0.f;
      for (int k = 0; k < x[0].size(); k++) {
        sum += x[i][k] * y[k][j];
      }
      row.push_back(sum);
    }
    ans.push_back(row);
  }
  return ans;
}

MT transpose(MT v) {
  MT ans;
  for (int i = 0; i < v[0].size(); i++) {
    VT row;
    for (int j = 0; j < v.size(); j++) {
      row.push_back(v[j][i]);
    }
    ans.push_back(row);
  }
  return ans;
}

MT inverse(MT v) {
  assert(v.size() == v[0].size());
  int n = v.size();
  MT I;
  for (int i = 0; i < n; i++) {
    VT row;
    for (int j = 0; j < n; j++) {
      if (i == j) {
        row.push_back(1);
      } else {
        row.push_back(0);
      }
    }
    I.push_back(row);
  }
  for (int i = 0; i < n; i++) {
    double max = abs(v[i][i]);
    int max_row = i;
    for (int j = 1 + i; j < n; j++) {
      if (v[j][i] > max) {
        max = abs(v[j][i]);
        max_row = j;
      }
    }
    if (max == 0) {
      cout << "this matrix got no inverse" << endl;
      return MT();
    }
    if (max_row != i) {
      swap(v[i], v[max_row]);
      swap(I[i], I[max_row]);
    }
    double tmp = v[i][i];
    for (int j = i; j < n; j++) {
      v[i][j] /= tmp;
    }
    for (int j = 0; j < n; j++) {
      I[i][j] /= tmp;
    }

    for (int j = 0; j < n; j++) {
      if (i != j) {
        tmp = v[j][i];
        for (int k = i; k < n; k++) {
          v[j][k] -= tmp * v[i][k];
        }
        for (int k = 0; k < n; k++) {
          I[j][k] -= tmp * I[i][k];
        }
      }
    }
  }
  cout << "inverse: \n";
  // print_matrix(I);
  return I;
}

double mean(VT v) {
  double sum = 0.f;
  accumulate(v.begin(), v.end(), sum);
  return sum / v.size();
}

MT covariance(MT m) {
  MT ans;
  int n = m.size();
  VT average(n, 0.f);
  for (VT v : m) {
    transform(v.cbegin(), v.cend(), average.cbegin(), average.begin(),
              plus<>());
  }
  for (auto &d : average) {
    d /= m.size();
  }
  for (VT &v : m) {
    transform(v.cbegin(), v.cend(), average.cbegin(), v.begin(),
              [=](double a, double b) { return (a - b) / (n - 1); });
  }

  ans = matmul(m, transpose(m));
  // print_matrix(ans);
  return ans;
}

void eigen(MT matrix) {
  // MT eigenMatrix(matrix.size(), VT(matrix[0].size(), 0.f));
  VT v;
  for (VT &vt : matrix) {
    for (double &d : vt) {
      v.push_back(d);
    }
  }
  // print_vector(v);
  MatrixXd A = Map<const MatrixXd>(v.data(), matrix[0].size(), matrix.size());
  A = A.transpose().eval();
  // 创建一个EigenSolver对象，用来对A进行特征向量求解
  EigenSolver<MatrixXd> solver(A);
  cout << A << endl;
  // 检查是否成功
  if (solver.info() == Success) {
    // 输出特征值
    std::cout << "The eigenvalues of A are: " << std::endl;
    std::cout << solver.eigenvalues() << std::endl;
    // 输出特征向量
    auto V = solver.eigenvectors();
    std::cout << "The eigenvectors of A are: " << std::endl;
    cout << V << endl;
    // 输出矩阵的元素
    // for (int i = 0; i < matrix.size(); i++) {
    //   for (int j = 0; j < matrix[0].size(); j++) {
    //     eigenMatrix[i][j] = V(i, j).real();
    //   }
    // }

    // std::cout << "The eigenMatrix: " << std::endl;
    // print_matrix(eigenMatrix);
    // data recovery
    // 选择前k个特征向量
    // int k = 3;
    // MatrixXd selectedEigenVectors = V.block(0, 0, V.rows(), k);

    // // 计算降维后的数据
    // MatrixXd reducedData = A * selectedEigenVectors;

    // // 计算恢复后的数据
    // MatrixXd recoveredData = reducedData * selectedEigenVectors.transpose();

  } else {
    // 输出错误信息
    std::cout << "Failed to compute the eigenvectors of A." << std::endl;
  }
}

int main(int argc, char *argv[]) {

  MT x = {{3, 1}, {7, 2}, {8, 3}, {4, 5}, {3, 6}};
  MT y = {{7, 4, 9, 6, 5}, {3, 7, 11}};
  auto xt = transpose(x);
  auto yt = transpose(y);

  // auto ans = matmul(xt, yt);
  // matmul(ans, inverse(ans));
  // auto ans = matmul(x, y);
  // cout << "original matrix: \n";
  // print_matrix(ans);
  // cout << "covariance matrix: \n";
  // auto ans_cov = covariance(ans);
  // eigen(ans_cov);
  // cout << "eigenVector is column vector of eigenMatrix\n";
  MatrixXd data(4, 2);
  data << 1.07, 2.16, 2.05, 4.1, -0.93, -2.03, -1.97, -4.08;
  // print_matrix(x);
  cout << data << endl;

  // auto cov1 = covariance(x);
  // 计算协方差矩阵
  MatrixXd cov = data.transpose() * data / (data.rows() - 1);

  // cout << "cov1\n";
  // print_matrix(cov1);
  cout << "cov\n" << cov << endl;
  // 计算特征值和特征向量
  Eigen::SelfAdjointEigenSolver<MatrixXd> eig(cov);
  MatrixXd eigenVectors = eig.eigenvectors();

  cout << "eigen\n" << eigenVectors << endl;
  // 选择前k个特征向量
  int k = 2;
  MatrixXd selectedEigenVectors =
      eigenVectors.block(0, 0, eigenVectors.rows(), k);

  cout << "selectedEigenVectors\n" << selectedEigenVectors << endl;
  // 计算降维后的数据
  MatrixXd reducedData = data * selectedEigenVectors;

  // 计算恢复后的数据
  MatrixXd recoveredData = reducedData * selectedEigenVectors.transpose();

  cout << "reducedData\n" << reducedData << endl;
  cout << "Original data:\n" << data << endl;
  cout << "Recovered data:\n" << recoveredData << endl;
  return 0;
}
