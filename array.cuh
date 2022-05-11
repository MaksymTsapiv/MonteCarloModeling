#ifndef MODEL_ARRAY_CUH
#define MODEL_ARRAY_CUH

#include "particle.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#define INDEX_OUT_OF_RANGE 1

class OrderedArray {
private:
    size_t size;
    size_t capacity;
    Particle* data;

public:
    OrderedArray(size_t capacity);
    ~OrderedArray();

    int insert(Particle value, size_t index);
    int remove(size_t index);

    const Particle* get_array();
    size_t getSize() const;

    void set_data(Particle* data, size_t size);
};

#endif //MODEL_ARRAY_CUH
