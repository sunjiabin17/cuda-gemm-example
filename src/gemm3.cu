#include <cuda_runtime.h>

#include <algorithm>
#include <random>
#include <vector>

template <typename T, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_M,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
          size_t BLOCK_TILE_SKEW_SIZE_X = 0U,
          size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(
    const T *A, size_t lda, const T *B, size_t ldb,
    T A_shared_memory_block[BLOCK_TILE_SIZE_M]
                           [BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
    T B_shared_memory_block[BLOCK_TILE_SIZE_K]
                           [BLOCK_TILE_SIZE_N + BLOCK_TILE_SKEW_SIZE_K],
    size_t shared_memory_block_tile_idx, size_t thread_linear_idx, size_t m,
    size_t n, size_t k) {

  // load A
  const size_t a_load_num =
      (BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) / NUM_THREADS;
  for (size_t load_idx = 0U; load_idx < a_load_num; ++load_idx) {
    const size_t A_shared_memory_block_row_idx =
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_K;
    const size_t A_shared_memory_block_col_idx =
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_K;

    const size_t A_row_idx =
        blockIdx.y * BLOCK_TILE_SIZE_M + A_shared_memory_block_row_idx;
    // [TODO] 这里为什么是shared_memory_block_tile_idx，而不是blockIdx.x;
    const size_t A_col_idx = shared_memory_block_tile_idx * BLOCK_TILE_SIZE_K +
                             A_shared_memory_block_col_idx;

    T val = static_cast<T>(0);
    if (A_row_idx < m and A_col_idx < k) {
      val = A[A_row_idx * lda + A_col_idx];
    }
    static_assert(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);

    A_shared_memory_block[A_shared_memory_block_row_idx]
                         [A_shared_memory_block_col_idx] = val;
  }

  // load B
  size_t b_load_num =
      (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N + NUM_THREADS - 1) / NUM_THREADS;
  for (size_t load_idx = 0U; load_idx < b_load_num; ++load_idx) {
    const size_t B_shared_memory_block_row_idx =
        (thread_linear_idx + load_idx * NUM_THREADS) / BLOCK_TILE_SIZE_N;
    const size_t B_shared_memory_block_col_idx =
        (thread_linear_idx + load_idx * NUM_THREADS) % BLOCK_TILE_SIZE_N;

    const size_t B_row_idx = shared_memory_block_tile_idx * BLOCK_TILE_SIZE_K +
                             B_shared_memory_block_row_idx;
    const size_t B_col_idx =
        blockIdx.x * BLOCK_TILE_SIZE_N + B_shared_memory_block_col_idx;

    T val = static_cast<T>(0);
    if (B_row_idx < k and B_col_idx < n) {
      val = B[B_row_idx * ldb + B_col_idx];
    }
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N % NUM_THREADS == 0U);

    B_shared_memory_block[B_shared_memory_block_row_idx]
                         [B_shared_memory_block_col_idx] = val;
  }
}

template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm_v3(size_t m, size_t n, size_t k, T alpha, const T *A,
                        size_t lda, const T *B, size_t ldb, T beta, T *C,
                        size_t ldc) {
  constexpr size_t NUM_THREADS = BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N;
  const size_t thread_linear_idx = blockIdx.y * blockDim.x + threadIdx.x;

  const size_t C_row_idx = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t C_col_idx = blockDim.x * blockIdx.x + threadIdx.x;

  __shared__ T A_shared_memory_block[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
  __shared__ T B_shared_memory_block[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

  const size_t num_shared_memory_block_tile =
      (k + BLOCK_TILE_SIZE_K - 1) / BLOCK_TILE_SIZE_K;

  size_t sum = static_cast<T>(0);
  for (size_t shared_memory_block_tile_idx = 0U;
       shared_memory_block_tile_idx < num_shared_memory_block_tile;
       ++shared_memory_block_tile_idx) {
    load_data_to_shared_memory<T, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M,
                               BLOCK_TILE_SIZE_K, NUM_THREADS>(
        A, lda, B, ldb, A_shared_memory_block, B_shared_memory_block,
        shared_memory_block_tile_idx, thread_linear_idx, m, n, k);
    __syncthreads();

    for (size_t k_idx = 0U; k_idx < BLOCK_TILE_SIZE_K; ++k_idx) {
      sum += A_shared_memory_block[threadIdx.y][k_idx] *
             B_shared_memory_block[k_idx][threadIdx.x];
    }
    __syncthreads();
  }
  if (C_row_idx < m and C_col_idx < n) {
    C[C_row_idx * ldc + C_col_idx] =
        alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
  }
}

template <typename T>
void launch_gemm_v3(size_t m, size_t n, size_t k, T alpha, const T *A,
                    size_t lda, const T *B, size_t ldb, T beta, T *C,
                    size_t ldc, cudaStream_t stream) {
  constexpr unsigned int BLOCK_TILE_SIZE_M = 32U;
  constexpr unsigned int BLOCK_TILE_SIZE_N = 32U;
  constexpr unsigned int BLOCK_TILE_SIZE_K = 32U;
  constexpr unsigned int NUM_THREADS = (BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_N);
  static_assert(BLOCK_TILE_SIZE_M * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
  static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_N % NUM_THREADS == 0U);
  const dim3 block_dim{BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_M, 1};
  const dim3 grid_dim{
      (static_cast<unsigned int>(n) + block_dim.x - 1) / block_dim.x,
      (static_cast<unsigned int>(m) + block_dim.y - 1) / block_dim.y, 1};
  gemm_v3<T, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K>
      <<<grid_dim, block_dim, 0, stream>>>(m, n, k, alpha, A, lda, B, ldb, beta,
                                           C, ldc);
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

  launch_gemm_v3(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);

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