#include <algorithm>
#include <boost/core/demangle.hpp>
#include <chrono>
#include <iostream>
#include <limits>
#include <random>

using namespace std;
using namespace std::chrono;
std::mt19937 gen(std::random_device{}());

template <class T> ostream &operator<<(ostream &o, vector<T> v) {
  for (T t : v) {
    o << t << ", ";
  }
  o << "\n";
  return o;
}

vector<double> safeSoftmax(vector<double> &v) {

  double m = *max_element(v.begin(), v.end());
  cout << "m = " << m << endl;
  double sum = accumulate(v.begin(), v.end(), sum,
                          [=](double a, double b) { return a + exp(b - m); });
  cout << "sum = " << sum << endl;
  transform(v.begin(), v.end(), v.begin(),
            [=](double a) { return (exp(a - m)) / sum; });
  return v;
}

vector<double> fastSoftmax(vector<double> &v) {
  int length = v.size();
  vector<double> ans(length, 0.f);
  double old_max = std::numeric_limits<double>::min();
  double new_max = std::numeric_limits<double>::min();
  double sum = 0.f;
  for (int i = 0; i < length; i++) {
    new_max = std::max(old_max, v[i]);
    sum = sum * exp(old_max - new_max) + exp(v[i] - new_max);
    old_max = new_max;
  }
  for (int i = 0; i < length; i++) {
    ans[i] = exp(v[i] - old_max) / sum;
  }
  cout << "m = " << new_max << endl;
  cout << "sum = " << sum << endl;
  return ans;
}

int main(int argc, char *argv[]) {
  std::normal_distribution d{0.f, 5.f};
  auto gen_double = [&d] { return d(gen); };
  vector<double> v;
  // for (double i = 0; i < 5; i++) {
  //   v.push_back(i - 2);
  // }
  // cout << v << softmax(v) << endl;
  // v.clear();
  int size = 200001;
  // double inteval = 0.1;
  // for (double i = 0; i < size; i++) {
  //   v.push_back(inteval * i - (size - 1) / 2 * inteval);
  // }
  // cout << v << softmax(v) << endl;
  // v.clear();
  for (double i = 0; i < size; i++) {
    v.push_back(gen_double());
  }

  vector<double> ve{v};

  auto s1 = high_resolution_clock::now();
  safeSoftmax(v);
  // cout << v << safeSoftmax(v) << endl;
  auto e1 = high_resolution_clock::now();
  auto s2 = high_resolution_clock::now();
  // cout << ve << fastSoftmax(ve) << endl;
  fastSoftmax(ve);
  auto e2 = high_resolution_clock::now();
  auto d1 = duration_cast<microseconds>(e1 - s1);
  auto d2 = duration_cast<microseconds>(e2 - s2);
  cout << "Time taken by function safeSoftmax : " << d1.count()
       << " microseconds" << endl;
  cout << "Time taken by function fastSoftmax : " << d2.count()
       << " microseconds" << endl;

  return 0;
}
