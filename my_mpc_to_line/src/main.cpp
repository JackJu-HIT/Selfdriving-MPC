#include <vector>
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "matplotlibcpp.h"
#include "MPC.h"
namespace plt = matplotlibcpp;

using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

int main() {
  //MPC mpc;
  int iters = 1;

  VectorXd ptsx(2);
  VectorXd ptsy(2);
  ptsx << -100, 100;
  ptsy << -1, -1;

  // The polynomial is fitted to a straight line so a polynomial with
  // order 1 is sufficient.
  auto coeffs = polyfit(ptsx, ptsy, 1);
}



