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

void OrderedArray::remove(size_t index) {
    if (index > size) {
        throw std::out_of_range("index out of range");
    }
    if (index == size - 1) {
        --size;
        return;
    }
    for (size_t i = index; i < size - 1; ++i) {
        data[i] = data[i + 1];
    }
    --size;
}

void OrderedArray::insert(Particle value, size_t index) {
    if (index > size) {
        throw std::out_of_range("index out of range");
    }
    for (size_t i = index; i < size; ++i) {
        data[i] = data[i + 1];
    }
    data[index] = value;
    ++size;
}

const Particle *OrderedArray::get_array() {
    return data;
}

size_t OrderedArray::getSize() {
    return size;
}
