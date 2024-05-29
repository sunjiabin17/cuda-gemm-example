#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <vector>

template <typename T>
__global__ void gemm_v1(size_t m, size_t n, size_t k, T alpha, const T *A,
                        size_t lda, const T *B, size_t ldb, T beta, T *C,
                        size_t ldc) {
  size_t C_row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  size_t C_col_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (C_row_idx < m and C_col_idx < n) {
    T sum = static_cast<T>(0);
    for (size_t k_idx = 0; k_idx < k; ++k_idx) {
      sum += A[C_row_idx * lda + k_idx] * B[k_idx * lda + C_col_idx];
    }
    C[C_row_idx * ldc + C_col_idx] = sum;
  }
}

template <typename T>
void launch_gemm_v2(size_t m, size_t n, size_t k, T alpha, const T *A,
                    size_t lda, const T *B, size_t ldb, T beta, T *C,
                    size_t ldc, cudaStream_t stream) {
  const dim3 block_dim{32, 32, 1};
  const dim3 grid_dim{
      (static_cast<unsigned int>(m) + block_dim.x - 1) / block_dim.x,
      (static_cast<unsigned int>(n) + block_dim.y - 1) / block_dim.y, 1};
  gemm_v1<T><<<grid_dim, block_dim, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb,
                                                 beta, C, ldc);
}

using T = float;

int main(int argc, char **argv) {
  size_t m = 1024;
  size_t n = 1024;
  size_t k = 1024;

  T *A = new T[m * k];
  T *B = new T[k * n];
  T *C = new T[m * n];
  T *C1 = new T[m * n];

  // set random seed
  // srand((unsigned)time(NULL));

  std::generate(A, A + m * k, []() { return (T)(rand() % 10); });
  std::generate(B, B + k * n, []() { return (T)(rand() % 10); });
  std::fill(C, C + m * n, 0.0f);

  T *dA, *dB, *dC;
  cudaMalloc(&dA, m * k * sizeof(T));
  cudaMalloc(&dB, k * n * sizeof(T));
  cudaMalloc(&dC, m * n * sizeof(T));

  cudaMemcpy(dA, A, m * k * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, k * n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);

  T alpha = 1.f;
  T beta = 0.f;

  size_t lda = k;
  size_t ldb = n;
  size_t ldc = n;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  launch_gemm_v2(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);

  // [TODO] something

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C1;

  return 0;
}