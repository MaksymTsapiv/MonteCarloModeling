#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

#define MAX_BLOCK_THREADS 512LU

#include "d3.cuh"
#include "patch.cuh"
#include "quat.cuh"

class Particle {
private:

public:
    double x, y, z, sigma;
    size_t id, clusterId;
    static size_t nextId;
    Quaternion quaternion{};
    std::vector<Patch> patches;

    Particle();
    Particle(double x_, double y_, double z_, double sigma_);
    Particle(double x_, double y_, double z_, double sigma_, Quaternion quaternion_, std::vector<Patch>  patches);

    __host__ __device__ D3<double> get_coord() const;
};

#endif //PARTICLE_FILE_H
