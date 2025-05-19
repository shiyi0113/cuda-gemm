#include <cute/tensor.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c)
{
    using namespace cute;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num / kNumElemPerThread)
    { // 未处理非对齐问题
        return;
    }

    Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
    Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
    Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));
    // 
    Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
    Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
    Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

    Tensor txR = make_tensor_like(txr);
    Tensor tyR = make_tensor_like(tyr);
    Tensor tzR = make_tensor_like(tzr);

    // LDG.128

    copy(txr, txR);
    copy(tyr, tyR);

    half2 a2 = {a, a};
    half2 b2 = {b, b};
    half2 c2 = {c, c};

    auto tzR2 = recast<half2>(tzR);
    auto txR2 = recast<half2>(txR);
    auto tyR2 = recast<half2>(tyR);

#pragma unroll
    for (int i = 0; i < size(tzR2); ++i)
    {
        // two hfma2 instruction
        tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
    }

    auto tzRx = recast<half>(tzR2);

    // STG.128
    copy(tzRx, tzr);
}

int main(int argc, char **argv)
{
    int num = 1024;
    thrust::host_vector<half> h_A(num);
    thrust::host_vector<half> h_B(num);
    thrust::host_vector<half> h_Z(num);
    for (int j = 0; j < num; ++j)
        h_A[j] = static_cast<half>(rand() / double(RAND_MAX)); // 生成0～1的随机数
    for (int j = 0; j < num; ++j)
        h_B[j] = static_cast<half>(rand() / double(RAND_MAX)); // 生成0～1的随机数
    for (int j = 0; j < num; ++j)
        h_Z[j] = static_cast<half>(-1);
    thrust::device_vector<half> d_A = h_A;
    thrust::device_vector<half> d_B = h_B;
    thrust::device_vector<half> d_Z = h_Z;
    dim3 Block(64);
    int num_tiles = num / 8;
    dim3 Gird((num_tiles + Block.x - 1) / Block.x);
    vector_add_local_tile_multi_elem_per_thread_half<8><<<Gird, Block>>>(
        thrust::raw_pointer_cast(d_Z.data()), // 获取 d_Z 的设备指针
        num,
        thrust::raw_pointer_cast(d_A.data()), // 获取 d_A 的设备指针
        thrust::raw_pointer_cast(d_B.data()), // 获取 d_B 的设备指针
        static_cast<half>(1.0f),              // 将 1.0f 转换为 half
        static_cast<half>(1.0f),              // 将 1.0f 转换为 half
        static_cast<half>(0.0f)               // 将 1.0f 转换为 half
    );
    thrust::host_vector<half> add_result = d_Z;
    for (int j = 0; j < 10; j++)
        printf("%f = %f + %f\n", __half2float(d_Z[j]), __half2float(d_A[j]), __half2float(d_B[j]));
}