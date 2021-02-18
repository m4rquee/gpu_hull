#include "../include/gpu_hull/gpu_vector.cuh"
#include <cassert>
#include <cuda.h>

int main() {
    typedef struct {
        float x;
        float y;
    } Point;

    const int size = 10000;
    auto harrayA = new int[size];
    auto harrayB = new Point[size];
    harrayA[0] = harrayA[size - 1] = -1;
    harrayB[0].x = harrayB[size - 1].y = -1;

    cudaStream_t streamA;
    cudaStreamCreateWithFlags(&streamA, CU_STREAM_NON_BLOCKING);
    cudaStream_t streamB;
    cudaStreamCreateWithFlags(&streamB, CU_STREAM_NON_BLOCKING);

    GPUVector<int> arrayA(size, streamA);
    arrayA.memset(0);
    GPUVector<Point> arrayB(size, streamB);
    arrayB.memset(0);

    arrayA.memcpy(harrayA);
    arrayB.memcpy(harrayB);

    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);
    assert(harrayA[0] == 0);
    assert(harrayA[size - 1] == 0);
    assert(harrayB[0].x == 0);
    assert(harrayB[size - 1].y == 0);

    arrayA.free();
    arrayB.free();
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    delete[] harrayA;
    delete[] harrayB;
    return 0;
}
