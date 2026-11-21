# CPU examples using the existing Ensign build configuration.
if(NOT CUDA_ENABLED)
  add_executable(matrix_test matrix_test.cpp)
  target_link_libraries(matrix_test PRIVATE Ensign)
endif()


if(NOT CUDA_ENABLED)
  add_executable(coefficients_test coefficients_test.cpp)
  target_link_libraries(coefficients_test PRIVATE Ensign)
endif()

if(NOT CUDA_ENABLED)
  add_executable(tensor_test tensor_test.cpp)
  target_link_libraries(tensor_test PRIVATE Ensign)
endif()

if(NOT CUDA_ENABLED)
  add_executable(vlasov_poisson1D_test vlasov_poisson1D_test.cpp)
  target_link_libraries(vlasov_poisson1D_test PRIVATE Ensign::Ensign)
endif()
