// From the repository root:
// cmake -S . -B build
// cmake --build build --target my_solver -j 2
// ./build/my-examples/my_solver

#include <cmath>
#include <iostream>
#include <random>
#include <generic/storage.hpp>
#include <generic/matrix.hpp>

int main() {
    using Ensign::multi_array;
    Ensign::Matrix::blas_ops blas;

    // Default storage and BLAS operations use the CPU.
    constexpr Index n = 3;
    multi_array<double, 2> A({n, n});
    multi_array<double, 2> B({n, n});
    multi_array<double, 2> C({n, n});

    // A fixed seed makes this example reproducible.
    std::mt19937 generator(1234);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (double& value : A) {
        value = uniform(generator);
    }
    for (double& value : B) {
        value = uniform(generator);
    }

    const multi_array<double, 2> sum = A + B;
    const multi_array<double, 2> difference = A - B;
    blas.matmul(A, B, C);

    // Right division X = A / B solves X * B = A.
    // Factor B^T once, then solve B^T * x_i^T = a_i^T for each row.
    Ensign::Matrix::lu_solver<double> solver(n);
    blas.transpose(B, solver.A);
    solver.lu();

    multi_array<double, 2> quotient({n, n});
    multi_array<double, 1> row({n});
    for (Index i = 0; i < n; ++i) {
        for (Index j = 0; j < n; ++j) {
            row(j) = A(i, j);
        }
        solver.solve(row);
        for (Index j = 0; j < n; ++j) {
            quotient(i, j) = row(j);
        }
    }


    std::cout << "A:\n";
    Ensign::print(A);
    std::cout << "B:\n";
    Ensign::print(B);
    std::cout << "A + B:\n";
    Ensign::print(sum);
    std::cout << "A - B:\n";
    Ensign::print(difference);
    std::cout << "C = A * B:\n";
    Ensign::print(C);
    std::cout << "X = A / B (X * B = A):\n";
    Ensign::print(quotient);

    multi_array<double, 2> reconstructed({n, n});
    blas.matmul(quotient, B, reconstructed);
    std::cout << "X * B = \n";
    Ensign::print(reconstructed);

    return 0;
}
