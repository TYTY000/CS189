#include <boost/core/demangle.hpp>
#include <iostream>
#include <vector>
using namespace std;
void print_vector(vector<double> u) {
  // cout << boost::core::demangle(typeid(u).name()) << " : " << endl;
  cout << "[";
  for (int i = 0; i < u.size(); i++) {
    cout << u[i];
    if (i < u.size() - 1) {
      cout << ", ";
    }
  }
  cout << "]" << endl;
}

void print_matrix(vector<vector<double>> A) {
  // cout << boost::core::demangle(typeid(A).name()) << " : " << endl;
  cout << "[ ";
  for (int i = 0; i < A.size(); i++) {
    if (i != 0)
      cout << "  ";
    print_vector(A[i]);
    // if (i < A.size() - 1) {
    //   cout << ", \n";
    // }
  }
  cout << " ]" << endl;
}
