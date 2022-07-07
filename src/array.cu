#include <iostream>
#include <stdexcept>
#include "array.cuh"

// this is helper function for debugging, it prints all elements in particles array
__global__ void print_kernel(Particle *particles, size_t size) {
    __syncthreads();
    for (int i = 0; i < size; i++) {
        printf("particle[%i]: %f %f %f %f\n", i, particles[i].x, particles[i].y,
                                particles[i].z, particles[i].sigma);
    }
    __syncthreads();
}

OrderedArray::OrderedArray(size_t capacity) {
    this->size = 0;
    this->capacity = capacity;
    cudaMalloc(&data, sizeof(Particle) * capacity);
}

OrderedArray::~OrderedArray() {
    cudaFree(data);
}

__global__ void get_particle_index_kernel(
                const Particle *particles, size_t particle_id, uint *index, size_t size)
{
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= size)
        return;

    if (particles[threadId].id == particle_id)
        *index = threadId;
}

int OrderedArray::update_particle(size_t id, Particle part) {
    uint *cudaIndex;
    cudaMalloc(&cudaIndex, sizeof(uint));

    int threadsPerBlock = MAX_BLOCK_THREADS;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    get_particle_index_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, id, cudaIndex, size);

    uint *index = new uint;
    cudaMemcpy(index, cudaIndex, sizeof(uint), cudaMemcpyDeviceToHost);

    if (*index > size) {
        std::cout << "INDEX OUT OF RANGE" << std::endl;
        return INDEX_OUT_OF_RANGE;
    }

    cudaMemcpy(&data[*index], &part, sizeof(Particle), cudaMemcpyHostToDevice);

    return 0;
}

int OrderedArray::remove_by_id(size_t id) {
    uint *cudaIndex;
    cudaMalloc(&cudaIndex, sizeof(uint));

    int threadsPerBlock = MAX_BLOCK_THREADS;
    size_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    get_particle_index_kernel<<<blocksPerGrid, threadsPerBlock>>>(data, id, cudaIndex, size);

    uint *index = new uint;
    cudaMemcpy(index, cudaIndex, sizeof(uint), cudaMemcpyDeviceToHost);

    auto res = remove(*index);

    cudaFree(cudaIndex);
    delete index;
    return res;
}

int OrderedArray::remove(size_t index) {
    if (index > size) {
        std::cout << "INDEX OUT OF RANGE" << std::endl;
        return INDEX_OUT_OF_RANGE;
    }

    if (index == size - 1) {
        --size;
        return 0;
    }

    auto parts_to_move = (size-(index+1));

    Particle *data_temp;
    cudaMalloc(&data_temp, sizeof(Particle)*parts_to_move);

    cudaMemcpy(data_temp, &data[index+1], parts_to_move*sizeof(Particle), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&data[index], data_temp, parts_to_move*sizeof(Particle), cudaMemcpyDeviceToDevice);

    --size;

    cudaFree(data_temp);
    return 0;
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

    cudaFree(data_temp);
    return 0;
}

const Particle *OrderedArray::get_array() {
    return data;
}

Particle *OrderedArray::get_mutable_array() {
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
