#include <algorithm>
#include <cmath>

#include <generic/matrix.hpp>
#include <generic/storage.hpp>
#include <generic/tensor.hpp>
#include <lr/coefficients.hpp>
#include <lr/lr.hpp>

class generator {
  private:
    Index i = 0;

  public:
    inline Index operator()() { return ++i; };
};

int main()
{
  std::cout << " ===================================== " << std::endl;
  std::cout << " Tensor for 3D multi_array testing " << std::endl;
  std::cout << " ===================================== " << std::endl;
  Index d0 = 2, d1 = 3, d2 = 4, d3 = 5;
  Ensign::multi_array<double, 3> ten({d0, d1, d2});
  Ensign::multi_array<double, 3> ten0({d0, d1, d2});
  Ensign::multi_array<double, 3> ten1({d0, d1, d2});
  Ensign::multi_array<double, 3> ten2({d0, d1, d2});
  Ensign::multi_array<double, 2> mat2({d0 * d1, d2}), mat2_ref({d0 * d1, d2});
  Ensign::multi_array<double, 2> mat1({d2 * d0, d1}), mat1_ref({d2 * d0, d1});
  Ensign::multi_array<double, 2> mat0({d1 * d2, d0}), mat0_ref({d1 * d2, d0});

  std::generate(std::begin(ten), std::end(ten), generator{});

  Ensign::Tensor::matricize<2>(ten, mat2);
  Ensign::Tensor::matricize<1>(ten, mat1);
  Ensign::Tensor::matricize<0>(ten, mat0);

  generator generator2{};
  for (Index k = 0; k < d2; ++k) {
      for (Index j = 0; j < d1; ++j) {
          for (Index i = 0; i < d0; ++i) {
              mat2_ref(i + j * d0, k) = generator2();
          }
      }
  }

  generator generator1{};
  for (Index k = 0; k < d2; ++k) {
      for (Index j = 0; j < d1; ++j) {
          for (Index i = 0; i < d0; ++i) {
              mat1_ref(k + i * d2, j) = generator1();
          }
      }
  }

  generator generator0{};
  for (Index k = 0; k < d2; ++k) {
      for (Index j = 0; j < d1; ++j) {
          for (Index i = 0; i < d0; ++i) {
              mat0_ref(j + k * d1, i) = generator0();
          }
      }
  }

  std::cout << "Ensign::Tensor::matricize<2>(ten, mat2), mat2 = \n";
  Ensign::print(mat2);
  std::cout << "Ensign::Tensor::matricize<1>(ten, mat1), mat1 = \n";
  Ensign::print(mat1);
  std::cout << "Ensign::Tensor::matricize<1>(ten, mat0), mat0 = \n";
  Ensign::print(mat0);

  std::cout << "mat2_ref = \n";
  Ensign::print(mat2_ref);
  std::cout << "mat1_ref = \n";
  Ensign::print(mat1_ref);
  std::cout << "mat0_ref = \n";
  Ensign::print(mat0_ref);

  Ensign::dump("tensor_ten.txt", ten);
  Ensign::dump("tensor_mat2.txt", mat2);
  Ensign::dump("tensor_mat1.txt", mat1);
  Ensign::dump("tensor_mat0.txt", mat0);
  Ensign::dump("tensor_mat2_ref.txt", mat2_ref);
  Ensign::dump("tensor_mat1_ref.txt", mat1_ref);
  Ensign::dump("tensor_mat0_ref.txt", mat0_ref);

  Ensign::Tensor::tensorize<2>(mat2, ten2);
  Ensign::Tensor::tensorize<1>(mat1, ten1);
  Ensign::Tensor::tensorize<0>(mat0, ten0);

  std::cout << " bool(ten2 == ten) = " << bool(ten2 == ten) << std::endl;
  std::cout << " bool(ten1 == ten) = " << bool(ten1 == ten) << std::endl;
  std::cout << " bool(ten0 == ten) = " << bool(ten0 == ten) << std::endl;

  /*
   * Generic matricization along the k-th axis (matricize<k>).
   * N_p is the size of axis p, and i_p is its zero-based index.
   *
   * \[
   * \begin{aligned}
   * \text{row} &= \sum_{\substack{1 \le p \le n \\ p \ne k}}
   *   i_p \prod_{\substack{1 \le q < p \\ q \ne k}} N_q, \\
   * \text{col} &= i_k.
   * \end{aligned}
   * \]
   */
  std::cout << " ===================================== " << std::endl;
  std::cout << " Tensor for 4D multi_array testing " << std::endl;
  std::cout << " ===================================== " << std::endl;
  Ensign::multi_array<double, 4> ten4({d0, d1, d2, d3}), ten4_0({d0, d1, d2, d3}),
      ten4_1({d0, d1, d2, d3}), ten4_2({d0, d1, d2, d3}), ten4_3({d0, d1, d2, d3});
  Ensign::multi_array<double, 2> mat4_3({d0 * d1 * d2, d3}), mat4_3_ref({d0 * d1 * d2, d3});
  Ensign::multi_array<double, 2> mat4_2({d0 * d1 * d3, d2}), mat4_2_ref({d0 * d1 * d3, d2});
  Ensign::multi_array<double, 2> mat4_1({d2 * d0 * d3, d1}), mat4_1_ref({d2 * d0 * d3, d1});
  Ensign::multi_array<double, 2> mat4_0({d1 * d2 * d3, d0}), mat4_0_ref({d1 * d2 * d3, d0});

  std::generate(std::begin(ten4), std::end(ten4), generator{});

  Ensign::Tensor::matricize<3>(ten4, mat4_3);
  Ensign::Tensor::matricize<2>(ten4, mat4_2);
  Ensign::Tensor::matricize<1>(ten4, mat4_1);
  Ensign::Tensor::matricize<0>(ten4, mat4_0);

  double begins = 1;
  for (Index w = 0; w < d3; ++w) {
    for (Index k = 0; k < d2; ++k) {
      for (Index j = 0; j < d1; ++j) {
        for (Index i = 0; i < d0; ++i) {
          mat4_3_ref(i + j * d0 + k * (d0*d1), w) = begins;
          mat4_2_ref(i + j * d0 + w * (d0*d1), k) = begins;
          mat4_1_ref(i + k * d0 + w * (d0*d2), j) = begins;
          mat4_0_ref(j + k * d1 + w * (d1*d2), i) = begins;
          begins += 1;
        }
      }
    }
  }

  std::cout << " bool(mat4_3 == mat4_3_ref) = " 
            << bool(mat4_3 == mat4_3_ref) << std::endl;
  std::cout << " bool(mat4_2 == mat4_2_ref) = " 
            << bool(mat4_2 == mat4_2_ref) << std::endl;
  std::cout << " bool(mat4_1 == mat4_1_ref) = " 
            << bool(mat4_1 == mat4_1_ref) << std::endl;
  std::cout << " bool(mat4_0 == mat4_0_ref) = " 
            << bool(mat4_0 == mat4_0_ref) << std::endl;
  
  Ensign::Tensor::tensorize<3>(mat4_3, ten4_3);
  Ensign::Tensor::tensorize<2>(mat4_2, ten4_2);
  Ensign::Tensor::tensorize<1>(mat4_1, ten4_1);
  Ensign::Tensor::tensorize<0>(mat4_0, ten4_0);

  std::cout << " bool(ten4_3 == ten4) = " << bool(ten4_3 == ten4) << std::endl;
  std::cout << " bool(ten4_2 == ten4) = " << bool(ten4_2 == ten4) << std::endl;
  std::cout << " bool(ten4_1 == ten4) = " << bool(ten4_1 == ten4) << std::endl;
  std::cout << " bool(ten4_0 == ten4) = " << bool(ten4_0 == ten4) << std::endl;

};