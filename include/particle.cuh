#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

#define MAX_BLOCK_THREADS 512LU

#include <vector>
#include <Eigen/Dense>
#include "d3.cuh"
#include "patch.cuh"
#include "quat.cuh"

class Particle {
public:
    double x, y, z, sigma;
    size_t id, clusterId, nPatches = 0;
    static size_t nextId;
    Quaternion quaternion{};
    std::vector<int> types{};
    Eigen::Matrix<double, Eigen::Dynamic, 3> db;

    Particle();
    Particle(double x_, double y_, double z_, double sigma_);
    Particle(double x_, double y_, double z_, double sigma_, Quaternion quaternion_, Eigen::Matrix<double, Eigen::Dynamic, 3> db_, std::vector<int> types_);

    __host__ __device__ D3<double> get_coord() const;
};

#endif //PARTICLE_FILE_H
