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

namespace gemm {

template <typename T>
void launch_gemm_v2(size_t m, size_t n, size_t k, const T* alpha, const T *A,
                    size_t lda, const T *B, size_t ldb, const T* beta, T *C,
                    size_t ldc, cudaStream_t stream) {
  const dim3 block_dim{32, 32, 1};
  const dim3 grid_dim{
      (static_cast<unsigned int>(m) + block_dim.x - 1) / block_dim.x,
      (static_cast<unsigned int>(n) + block_dim.y - 1) / block_dim.y, 1};
  gemm_v1<T><<<grid_dim, block_dim, 0, stream>>>(m, n, k, *alpha, A, lda, B, ldb,
                                                 *beta, C, ldc);         
}

template void launch_gemm_v2<float>(size_t m, size_t n, size_t k, const float* alpha,
                                    const float *A, size_t lda, const float *B,
                                    size_t ldb, const float* beta, float *C, size_t ldc,
                                    cudaStream_t stream);

template void launch_gemm_v2<double>(size_t m, size_t n, size_t k, const double* alpha,
                                     const double *A, size_t lda,
                                     const double *B, size_t ldb, const double* beta,
                                     double *C, size_t ldc, cudaStream_t stream);

} // namespace gemm