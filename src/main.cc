#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#define PRINT_MATRIX(m, n, A) \
  do {                         \
    for (size_t i = 0; i < m; i++) { \
      for (size_t j = 0; j < n; j++) { \
        std::cout << A[i * n + j] << " "; \
      } \
      std::cout << std::endl; \
    } \
    std::cout << std::endl; \
  } while (0)


namespace gemm {
template <typename T>
void launch_gemm_v1(size_t m, size_t n, size_t k, const T* alpha, const T *A,
                    size_t lda, const T *B, size_t ldb, const T* beta, T *C,
                    size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_v2(size_t m, size_t n, size_t k, const T* alpha, const T *A,
                    size_t lda, const T *B, size_t ldb, const T* beta, T *C,
                    size_t ldc, cudaStream_t stream);

template <typename T>
void launch_gemm_v3(size_t m, size_t n, size_t k, const T* alpha, const T *A,
                    size_t lda, const T *B, size_t ldb, const T* beta, T *C,
                    size_t ldc, cudaStream_t stream);


void launch_cublas_gemm(size_t m, size_t n, size_t k, const float* alpha, const float *A,
                        size_t lda, const float *B, size_t ldb, const float* beta, float *C,
                        size_t ldc) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
  cublasDestroy(handle);
}

}  // namespace gemm

bool compare(const float *A, const float *B, const size_t& size) {
  for (size_t i = 0; i < size; i++) {
    if (std::abs(A[i] - B[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}


using T = float;

int main(int argc, char **argv) {
  size_t m = 2048;
  size_t n = 1024;
  size_t k = 512;

  T *A = new T[m * k];
  T *B = new T[k * n];
  T *C = new T[m * n];
  T *C1 = new T[m * n];

  // set random seed
  // srand((unsigned)time(NULL));

  std::generate(A, A + m * k, []() { return (T)(rand() % 10); });
  std::generate(B, B + k * n, []() { return (T)(rand() % 10); });
  std::fill(C, C + m * n, 0.0f);
  std::fill(C1, C1 + m * n, 0.0f);

  T *dA, *dB, *dC, *dC1;
  cudaMalloc(&dA, m * k * sizeof(T));
  cudaMalloc(&dB, k * n * sizeof(T));
  cudaMalloc(&dC, m * n * sizeof(T));
  cudaMalloc(&dC1, m * n * sizeof(T));

  cudaMemcpy(dA, A, m * k * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, k * n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dC1, C1, m * n * sizeof(T), cudaMemcpyHostToDevice);

  T alpha = 1.f;
  T beta = 0.f;

  size_t lda = k;
  size_t ldb = n;
  size_t ldc = n;

  gemm::launch_cublas_gemm(m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC1, ldc);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  std::cout << "launch gemm" << std::endl;
  gemm::launch_gemm_v3(m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc, stream);

  // compare
  cudaMemcpy(C, dC, m * n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(C1, dC1, m * n * sizeof(T), cudaMemcpyDeviceToHost);

  // PRINT_MATRIX(m, k, A);
  // PRINT_MATRIX(k, n, B);
  // PRINT_MATRIX(m, n, C);
  // PRINT_MATRIX(m, n, C1);


  if (compare(C, C1, m * n)) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC1);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C1;

  return 0;
}