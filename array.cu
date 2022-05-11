#include <stdexcept>
#include "array.cuh"


OrderedArray::OrderedArray(size_t capacity) {
    this->size = 0;
    this->capacity = capacity;
    cudaMalloc(&data, sizeof(Particle) * capacity);
}

OrderedArray::~OrderedArray() {
    cudaFree(data);
}

int OrderedArray::remove(size_t index) {
    if (index > size) {
        return INDEX_OUT_OF_RANGE;
    }
    if (index == size - 1) {
        --size;
        return 0;
    }
    for (size_t i = index; i < size - 1; ++i) {
        data[i] = data[i + 1];
    }
    --size;

    return 0;
}

// this is helper function for debugging, it prints all elements in particles array
__global__ void print_kernel(Particle *particles, size_t size) {
    for (int i = 0; i < size; i++) {
        printf("particle[%i]: %f %f %f %f\n", i, particles[i].x, particles[i].y,
                                particles[i].z, particles[i].sigma);
    }
}

int OrderedArray::insert(Particle value, size_t index) {
    if (index > size) {
        return INDEX_OUT_OF_RANGE;
    }

    auto parts_to_move = (size-index);

    Particle *data_temp;
    cudaMalloc(&data_temp, sizeof(Particle)*parts_to_move);

    cudaMemcpy(data_temp, &data[index], parts_to_move*sizeof(Particle), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&data[index+1], data_temp, parts_to_move*sizeof(Particle), cudaMemcpyDeviceToDevice);

    cudaMemcpy(&data[index], &value, sizeof(Particle), cudaMemcpyHostToDevice);

    ++size;
    return 0;
}

const Particle *OrderedArray::get_array() {
    return data;
}

size_t OrderedArray::getSize() const {
    return size;
}

void OrderedArray::set_data(Particle *data, size_t size) {
    if (size > capacity) {
        throw std::runtime_error("Something went wrong when setting OrderedArray on GPU:\
                size of Particle array is greater than capacity.");
    }
    cudaFree(this->data);
    cudaMalloc(&this->data, sizeof(Particle) * capacity);
    cudaMemcpy(this->data, data, sizeof(Particle) * size, cudaMemcpyHostToDevice);

    this->size = size;
}
