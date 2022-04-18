#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

#include "d3.cuh"
class Particle {
private:

public:
    double x, y, z, sigma;
    size_t id;
    static size_t nextId;

    Particle();
    Particle(double x_, double y_, double z_, double sigma_);

    __host__ __device__ D3<double> get_coord() const;
};

#endif //PARTICLE_FILE_H
