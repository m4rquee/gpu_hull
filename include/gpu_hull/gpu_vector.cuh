#ifndef GPU_HULL_GPU_VECTOR_CUH
#define GPU_HULL_GPU_VECTOR_CUH

#include <driver_types.h>

template<typename T>
class GPUVector {
    size_t size;
    cudaStream_t stream;// operations stream
    T *vector;

public:
    GPUVector(size_t size, cudaStream_t stream);
    ~GPUVector();
    void free();
    void memset(int value, size_t count);
    void memset(int value);
    void memcpy(T *dst);
};

template<typename T>
GPUVector<T>::GPUVector(size_t size, cudaStream_t stream) : size(size), stream(stream) {
    cudaMallocAsync((void **) &this->vector, this->size * sizeof(T), this->stream);
}

template<typename T>
GPUVector<T>::~GPUVector() {
    if (this->vector != nullptr)
        cudaFreeAsync(this->vector, this->stream);
}

template<typename T>
void GPUVector<T>::free() {
    cudaFreeAsync(this->vector, this->stream);
    this->vector = nullptr;
}

template<typename T>
void GPUVector<T>::memset(int value, size_t count) {
    cudaMemsetAsync(this->vector, value, count, this->stream);
}

template<typename T>
void GPUVector<T>::memset(int value) {
    memset(value, this->size * sizeof(int));
}

template<typename T>
void GPUVector<T>::memcpy(T *dst) {
    cudaMemcpyAsync(dst, this->vector, this->size * sizeof(T), cudaMemcpyDeviceToHost, this->stream);
}

#endif//GPU_HULL_GPU_VECTOR_CUH
