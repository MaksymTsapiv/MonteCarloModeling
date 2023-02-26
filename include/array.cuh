#ifndef MODEL_ARRAY_CUH
#define MODEL_ARRAY_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#include "particle.cuh"
#include "d3.cuh"

#define INDEX_OUT_OF_RANGE 1

class OrderedArray {
private:
    size_t size;
    size_t capacity;
    Particle* data{};

public:
    explicit OrderedArray(size_t capacity);
    ~OrderedArray();

    int insert(Particle value, size_t index);
    int remove(size_t index);
    int remove_by_id(size_t id);
    int update_particle(size_t id, Particle part);

    const Particle* get_array();
    Particle* get_mutable_array();
    size_t getSize() const;

    void set_data(Particle* data, size_t size);
};

#endif //MODEL_ARRAY_CUH
