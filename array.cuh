#ifndef MODEL_ARRAY_CUH
#define MODEL_ARRAY_CUH

#include "particle.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

class OrderedArray {
private:
    size_t size;
    size_t capacity;
    Particle* data;

public:
    OrderedArray(size_t capacity);
    ~OrderedArray();

    void insert(Particle value, size_t index);
    void remove(size_t index);

    const Particle* get_array();
    size_t getSize();
};

#endif //MODEL_ARRAY_CUH
