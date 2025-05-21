#include <cute/tensor.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cublas_v2.h>

template <typename LayoutType>
bool save_latex_to_file(const LayoutType &layout, const std::string &filename)
{
    // 创建要捕获输出的文件
    FILE *file = fopen(filename.c_str(), "w");
    if (!file)
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    // 保存原始的stdout
    int original_stdout = dup(fileno(stdout));

    // 重定向stdout到文件
    dup2(fileno(file), fileno(stdout));

    // 调用print_latex，现在应该输出到文件
    print_latex(layout);

    // 冲洗缓冲区
    fflush(stdout);

    // 恢复原始stdout
    dup2(original_stdout, fileno(stdout));
    close(original_stdout);

    // 关闭文件
    fclose(file);

    std::cout << "已尝试保存LaTeX到文件: " << filename << std::endl;
    return true;
}

template <typename TA, typename TB, typename TC, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_cute_naive(TC *Cptr, const TA *Aptr, const TB *Bptr, int m, int n, int k)
{
    using namespace cute;
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));  //(m,k):(k,1) //(81920,256):(256,1)
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));  //(n,k):(k,1) //(256  ,256):(256,1) 
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));  //(m,n):(n,1) //(81920,256):(256,1)

    int ix = blockIdx.x;
    int iy = blockIdx.y;

    Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
    Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
    //  gA(kTileM, kTileK, num_tile_k) //(128,64 ,4):(256,1,64)
    //  gB(kTileN, kTileK, num_tile_k) //(128,64 ,4):(256,1,64)
    //  gC(kTileM, kTileN)             //(128,128  ):(256,1) 

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tAgA = thr_mma.partition_A(gA); // (MMA, MMA_M, MMA_K, num_tile_k)  ((_2,_2,_2),_8 ,_4 ,4):((_1,2048,_8),4096,_16,_64)
    auto tBgB = thr_mma.partition_B(gB); // (MMA, MMA_N, MMA_K, num_tile_k)  ((_2,_2)   ,_16,_4 ,4):((_1,_8),2048,_16,_64)
    auto tCgC = thr_mma.partition_C(gC); // (MMA, MMA_M, MMA_N)              ((_2,_2)   ,_8 ,_16)  :((_1,2048),4096,_8)

    auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)  ((_2,_2,_2),_8 ,_4 ):((_1,_2,_4),_32,_8)
    auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)  ((_2,_2)   ,_16,_4 ):((_1,_2)   ,_16,_4)
    auto tCrC = thr_mma.partition_fragment_C(gC(_, _));    // (MMA, MMA_M, MMA_N)  ((_2,_2)   ,_8 ,_16):((_1,_2)   ,_4 ,_32)

    clear(tCrC);

    int num_tile_k = size<2>(gA);

    for (int itile = 0; itile < num_tile_k; ++itile)
    {
        cute::copy(tAgA(_, _, _, itile), tArA);        // g->r
        cute::copy(tBgB(_, _, _, itile), tBrB);        // g->r
        cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC); // gemm   C = A * B + C
    }

    cute::copy(tCrC, tCgC);  // r->g
}

int main()
{
    using namespace cute;
    const int m = 81920;
    const int n = 256;
    const int k = 256;
    using TA = half;
    using TB = half;
    using TC = half;
    const int kTileN = 128;
    const int kTileM = 128;
    const int kTileK = 64;

    thrust::host_vector<TA> h_A(m * k);
    thrust::host_vector<TB> h_B(k * n);
    thrust::host_vector<TC> h_C(m * n);

    for (int j = 0; j < m * k; ++j)
        h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int j = 0; j < k * n; ++j)  
        h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
    for (int j = 0; j < m * n; ++j)
        h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    using mma_op = cute::SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = cute::MMA_Traits<mma_op>;
    using mma_atom = cute::MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        make_layout(Shape<_1, _1, _1>{}),
                                        Tile<_16,_8,_16>{}));

    save_latex_to_file(MMA{},"MMA"); 
    dim3 block(size(MMA{}));
    dim3 grid(n / kTileN, m / kTileM);
    gemm_cute_naive<TA, TB, TC, kTileM, kTileN, kTileK, MMA><<<grid, block>>>(
        thrust::raw_pointer_cast(d_C.data()),
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        m, n, k);
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> result_cute = d_C;

    // cublas
    thrust::device_vector<TC> d_Z = h_C;

    cublasHandle_t handle;
    cublasCreate(&handle);

    half alpha = half(1.f);
    half beta = half(0.f);
    cublasStatus_t ret = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     n, m, k,
                                     &alpha,
                                     thrust::raw_pointer_cast(d_B.data()), k,
                                     thrust::raw_pointer_cast(d_A.data()), k,
                                     &beta,
                                     thrust::raw_pointer_cast(d_Z.data()), n);
    if (ret != CUBLAS_STATUS_SUCCESS)
    {
        printf("blas err = %d, str = %s\n", ret, cublasGetStatusString(ret));
    }
    CUTE_CHECK_LAST();
    thrust::host_vector<TC> result_cublas = d_Z;
    // compare
    float threshold = 0.1;
    for (int i = 0; i < m * n; ++i)
    {
        float v1 = result_cute[i];
        float v2 = result_cublas[i];
        if (fabs(v2 - v1) > threshold)
        {
            printf("error %d: v1 = %f, v2 = %f\n", i, v1, v2);
            break;
        }
    }
}