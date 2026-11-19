#include <generic/matrix.hpp>
#include <generic/storage.hpp>

#include <lr/coefficients.hpp>
#include <Eigen/Dense>

#include <random>

int main() {
  using namespace Ensign;
  using namespace Ensign::Matrix;

  blas_ops blas;

  // A = a;
  // B = b;
  // S = diag(co);
  // OUT = A^T * S * B;
  int r = 2;
  int ca = 4;
  int cb = 3;
  double co = 0.3;

  multi_array<double,2> a({r,ca}), b({r,cb}), out({ca,cb}), out2({ca,cb});
  using ColMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  Eigen::Map<ColMajorMatrix> A(a.data(), a.shape()[0], a.shape()[1]);
  Eigen::Map<ColMajorMatrix> B(b.data(), b.shape()[0], b.shape()[1]);
  Eigen::MatrixXd S = Eigen::MatrixXd(r, r);
  S.setZero();
  for (int i = 0; i < r; i++) { S(i, i) = co; }
  
  std::mt19937 generator(783);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (double& value : a) {
      value = uniform(generator);
  }
  for (double& value : b) {
      value = uniform(generator);
  }


  std::cout << "a:\n";
  Ensign::print(a);
  std::cout << "A (Eigen::Map):\n" << A << "\n\n";
  std::cout << "b:\n";
  Ensign::print(b);

  Eigen::MatrixXd out_eigen = A.transpose() * S * B;
  std::cout << "out_eigen:\n" << out_eigen << "\n";

  coeff(a, b, co, out, blas);

  int rows_a = a.shape()[0];
  int cols_a = a.shape()[1];
  int rows_b = b.shape()[0];
  int cols_b = b.shape()[1];
  std::cout << "rows_a = " << rows_a << ", cols_a = " << cols_a << std::endl;

  double* ptr_j;
  double* ptr_l;
  set_zero(out2);
  for(int j = 0; j < cols_a; j++){
    for(int l = 0; l < cols_b; l++){
      ptr_j = a.extract({j});
      ptr_l = b.extract({l});
      for(int i = 0; i < rows_a; i++){
        out2(j,l) += ptr_j[i] * ptr_l[i] * co;
      }
    }
  }

  std::cout << "out:\n";
  Ensign::print(out);
  std::cout << "out2:\n";
  Ensign::print(out2);

  
  return 0;
}