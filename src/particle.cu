// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <cstddef>
#include <utility>
#include "particle.cuh"

size_t Particle::nextId = 0;

Particle::Particle() : x(0), y(0), z(0), sigma(0), id(nextId), clusterId(nextId) { nextId++; }

Particle::Particle(double x_, double y_, double z_, double sigma_) :
        x(x_), y(y_), z(z_), sigma(sigma_), id(nextId), clusterId(nextId) { ++nextId;}


Particle::Particle (double x_, double y_, double z_, double sigma_, Quaternion quaternion_,
        Eigen::Matrix<double, Eigen::Dynamic, 3> db_, std::vector<int> types_) :
    x(x_), y(y_), z(z_), sigma(sigma_), id(nextId), clusterId(nextId),
    quaternion(quaternion_), db(db_), types(types_)
{ nextId++; }


__host__ __device__ D3<double> Particle::get_coord() const {
    return {x, y, z};
}
