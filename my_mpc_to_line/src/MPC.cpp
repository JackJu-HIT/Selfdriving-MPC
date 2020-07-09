#include "MPC.h"
#include <math.h>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "matplotlibcpp.h"
//#include "helpers.h"
#include "Eigen-3.3/Eigen/QR"
using CppAD::AD;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;
namespace plt = matplotlibcpp;
//为输出画图做准备，记录每次预测25个步长的数据。
vector<double> x_sum;
vector<double> y_sum;
vector<double> psi_sum;
vector<double> v_sum;
vector<double> cte_sum;
vector<double> epsi_sum;
vector<double> delta_sum;
vector<double> a_sum;
//namespace plt = matplotlibcpp;
// We set the number of timesteps to 25
// and the timestep evaluation frequency or evaluation
// period to 0.05.
size_t N = 25;//**预测步长*
double dt = 0.05;//采样时间
// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Both the reference cross track and orientation errors are 0.
// The reference velocity is set to 40 mph.
double ref_v = 40;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
//求解器将所有状态变量和执行器变量合并为一个奇异矢量。 因此，我们应该确定一个变量何时开始而另一个变量何时结束，以使我们的生活更轻松。
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;//这几行代码的目的是力求把上述数值放到一个向量中，定义的索引。

class FG_eval {
 public:
  VectorXd coeffs;
  // Coefficients of the fitted polynomial.
  FG_eval(VectorXd coeffs) { this->coeffs = coeffs; }//这句是c++中的构造函数。由对象传递参数。

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  // `fg` is a vector containing the cost and constraints.
  // `vars` is a vector containing the variable values (state & actuators).
  void operator()(ADvector& fg, const ADvector& vars) {
    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // The part of the cost based on the reference state.成本的一部分基于参考状态。
    for (int t = 0; t < N; ++t) {
      fg[0] += CppAD::pow(vars[cte_start + t], 2);
      fg[0] += CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Minimize the use of actuators.
    for (int t = 0; t < N - 1; ++t) {
      fg[0] += CppAD::pow(vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (int t = 0; t < N - 2; ++t) {
      fg[0] += CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    //
    // Setup Constraints
    //注意：在本节中，您将设置模型约束。
//初始约束
 //由于成本位于fg的索引0处，因此我们将每个起始索引加1。 这增加了所有其他值的位置。
    // NOTE: In this section you'll setup the model constraints.

    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (int t = 1; t < N; ++t) {
      // The state at time t+1 .
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // The state at time t.
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      AD<double> f0 = coeffs[0] + coeffs[1] * x0; //因为是直线。
      AD<double> psides0 = CppAD::atan(coeffs[1]);

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psides[t] + v[t] * delta[t] / Lf * dt
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);//我的理解就是改成拉格朗日乘数法形式。
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] =
          cte1 - ((f0 - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] =
          epsi1 - ((psi0 - psides0) + v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition
//

MPC::MPC() {}
MPC::~MPC() {}

std::vector<double> MPC::Solve(const VectorXd &x0, const VectorXd &coeffs) {
  typedef CPPAD_TESTVECTOR(double) Dvector;

  double x = x0[0];
  double y = x0[1];
  double psi = x0[2];
  double v = x0[3];
  double cte = x0[4];
  double epsi = x0[5];

  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints = N * 6;

  // Initial value of the independent variables.
  // Should be 0 except for the initial values.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; ++i) {
    vars[i] = 0.0;
  }
  // Set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // Lower and upper limits for x
  Dvector vars_lowerbound(n_vars);//定义
  Dvector vars_upperbound(n_vars);

  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.将所有非执行器的上限和下限设置为最大负值和正值
  //。这句话的意思就是除了执行器delta和a，剩下的位移速度等上下界设为最大。
  for (int i = 0; i < delta_start; ++i) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (int i = delta_start; i < a_start; ++i) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (int i = a_start; i < n_vars; ++i) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }

  // Lower and upper limits for constraints
  // All of these should be 0 except the initial
  // state indices.约束的上限和下限除初始状态索引外，所有这些都应为0。
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; ++i) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
   /***待考虑，为何要赋予初始值******/ 
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;
 
//从这步开始为求最优解做准备。
  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // options
  std::string options;
  options += "Integer print_level  0\n";//您可以使用带有以下语法的行来设置任何Ipopt整数选项：整数 名称值  名称是任何有效的Ipopt整数选项，值是其设置。
  options += "Sparse  true        forward\n";  //雅可比集将使用SparseJacobianForward计算。
  options += "Sparse  true        reverse\n";//雅可比集将使用SparseJacobianReverse计算。

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem、、https://www.coin-or.org/CppAD/Doc/ipopt_solve.htm
  /**The function ipopt::solve solves nonlinear programming problems of the form
   minimize f(x) 
   subject to gl≤g(x)≤gu xl≤x≤xu
****/
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  //
  // Check some of the solution values
  //
  bool ok = true;
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;
  
  for(int i=0;i<y_start;i++)
  {std::cout<<"打印第"<<i<<"个x:"<<solution.x[i]<<" ";
 x_sum.push_back(solution.x[i]);
}
  std::cout<<std::endl;
  for(int i=y_start;i<psi_start;i++)
  {
std::cout<<"打印第"<<i<<"个y:"<<solution.x[i]<<" ";
y_sum.push_back(solution.x[i]);
  }
  
   std::cout<<std::endl;
   for(int i=psi_start;i<v_start;i++)
   {
std::cout<<"打印第"<<i<<"个psi:"<<solution.x[i]<<" ";
psi_sum.push_back(solution.x[i]);
   }
  
   std::cout<<std::endl;
for(int i=v_start;i<cte_start;i++)
{
std::cout<<"打印第"<<i<<"个v:"<<solution.x[i]<<" ";
v_sum.push_back(solution.x[i]);
}
  
   std::cout<<std::endl;
   for(int i=cte_start;i<epsi_start;i++)
   {
 std::cout<<"打印第"<<i<<"个cte:"<<solution.x[i]<<" ";
 cte_sum.push_back(solution.x[i]);
   }

   std::cout<<std::endl;
for(int i=epsi_start;i<delta_start;i++)
{
 std::cout<<"打印第"<<i<<"个epsi:"<<solution.x[i]<<" ";
 epsi_sum.push_back(solution.x[i]);
}
 
   std::cout<<std::endl;
for(int i=delta_start;i<a_start;i++)
{
std::cout<<"打印第"<<i<<"个delta:"<<solution.x[i]<<" ";
delta_sum.push_back(solution.x[i]);
}
   std::cout<<std::endl;
   for(int i=a_start;i<n_vars;i++)
   {
 std::cout<<"打印第"<<i<<"个a:"<<solution.x[i]<<" ";
 a_sum.push_back(solution.x[i]);
   }
   std::cout<<std::endl;
 //plt::title("x");
 // plt::plot(x_sum);
 //matplotlibcpp::title("x");
  //matplotlibcpp::plot(x_sum);
/*
  plt::title("CTE");
  plt::plot(test);
*/
  // matplotlibcpp::show();

  auto cost = solution.obj_value;

  std::cout << "Cost " << cost << std::endl;
  //return x_sum,y_sum;
  return y_sum;
/*
  return {solution.x[x_start + 1],   solution.x[y_start + 1],
          solution.x[psi_start + 1], solution.x[v_start + 1],
          solution.x[cte_start + 1], solution.x[epsi_start + 1],
          solution.x[delta_start],   solution.x[a_start]};
          */
        
}


/****曲线拟合********/
VectorXd polyfit(const VectorXd &xvals, const VectorXd &yvals, int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);

  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); ++i) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); ++j) {
    for (int i = 0; i < order; ++i) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);

  return result;
}
// Evaluate a polynomial.
double polyeval(const VectorXd &coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); ++i) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

/*****主程序*******/


int main() {
  
  MPC mpc;
  int iters = 2;

  VectorXd ptsx(2);
  VectorXd ptsy(2);
  ptsx << -100, 100;
  ptsy << -1, -1;

  // The polynomial is fitted to a straight line so a polynomial with
  // order 1 is sufficient.
  auto coeffs = polyfit(ptsx, ptsy, 1);

  // NOTE: free feel to play around with these
  double x = -1;
  double y = 10;
  double psi = 0;
  double v = 10;
  // The cross track error is calculated by evaluating at polynomial at x, f(x)
  // and subtracting y.
  double cte = polyeval(coeffs, x) - y;
  // Due to the sign starting at 0, the orientation error is -f'(x).
  // derivative of coeffs[0] + coeffs[1] * x -> coeffs[1]
  double epsi = psi - atan(coeffs[1]);

  VectorXd state(6);
  state << x, y, psi, v, cte, epsi;

  vector<double> x_vals = {state[0]};
  vector<double> y_vals = {state[1]};
  vector<double> psi_vals = {state[2]};
  vector<double> v_vals = {state[3]};
  vector<double> cte_vals = {state[4]};
  vector<double> epsi_vals = {state[5]};
  vector<double> delta_vals = {};
  vector<double> a_vals = {};
   vector<double>test;
   vector<double>test2;

  for (size_t i = 0; i < iters; ++i) {
    cout << "Iteration " << i << endl;
 //auto vars = mpc.Solve(state, coeffs);
  auto vars= mpc.Solve(state, coeffs);
   state << vars[0], vars[1], vars[2], vars[3], vars[4], vars[5];
  // auto vars = mpc.Solve(state, coeffs);
//保存预测步长范围状态及输入，用于绘图**
    
    //x_vals.push_back(vars[0]);
    //y_vals.push_back(vars[1]);
   // for(int i=0;i<25;i++)
    //cout<<"new打印第"<<i<<"个x:"<<x_vals[i]<<" ";
   /* psi_vals.push_back(vars[2]);
    v_vals.push_back(vars[3]);
    cte_vals.push_back(vars[4]);
    epsi_vals.push_back(vars[5]);
    
    delta_vals.push_back(vars[6]);
    a_vals.push_back(vars[7]);

    state << vars[0], vars[1], vars[2], vars[3], vars[4], vars[5];
    cout << "x = " << vars[0] << endl;
    cout << "y = " << vars[1] << endl;
    cout << "psi = " << vars[2] << endl;
    cout << "v = " << vars[3] << endl;
    cout << "cte = " << vars[4] << endl;
    cout << "epsi = " << vars[5] << endl;
    cout << "delta = " << vars[6] << endl;
    cout << "a = " << vars[7] << endl;
    cout << endl;
 */
   // test=vars;
   // test2.push_back(vars[1]);
  //}

  // Plot values
  // NOTE: feel free to play around with this.
  // It's useful for debugging!
/*
  plt::subplot(4, 1, 1);
  plt::title("CTE");
  plt::plot(cte_vals);
  plt::subplot(4, 1, 2);
  plt::title("Delta (Radians)");
  plt::plot(delta_vals);
  plt::subplot(4, 1, 3);
  plt::title("Velocity");
  plt::plot(v_vals);
  */
  //plt::subplot(4,2,1);
  //plt::title("x");
  //plt::plot(test);
  //plt::subplot(4,2,2);
  //plt::title("y");
  //plt::plot(test2);
  
  }


  /**预测的位置信息**/
  plt::subplot(4,2,1);
  plt::title("x of prediction");
  plt::plot(x_sum);
  //plt::xlabel("x");
  plt::ylabel("y");
 
  //航向信息
  plt::subplot(4,2,2);
   plt::title("psi prediction");
  plt::plot(psi_sum);
  plt::ylabel("psi");

  //速度信息
   plt::subplot(4,2,3);
  plt::title("v prediction");
  plt::plot(v_sum);
  plt::ylabel("v");

  //cte
plt::subplot(4,2,4);
  plt::title("cte error prediction");
  plt::plot(cte_sum);
  plt::ylabel("cte");

  //epsi
 plt::subplot(4,2,5);
  plt::title("epsi prediction");
  plt::plot(epsi_sum);
  plt::ylabel("epsi");

  //delta(控制信息)
   plt::subplot(4,2,6);
  plt::title("delta control of prediction");
  plt::plot(delta_sum);
  plt::ylabel("delta");

  //a(控制信息)
  plt::subplot(4,2,7);
  plt::title("a control of prediction");
  plt::plot(a_sum);
  plt::ylabel("a");


plt::subplot(4,2,8);
  plt::title("y of prediction");
  plt::plot(y_sum);
  //plt::xlabel("x");
  plt::ylabel("y");



 plt::show();
}