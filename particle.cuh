#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

class Particle {
private:
    double x_cor, y_cor, z_cor, sigma;
    size_t id;

public:
    static size_t nextId;

    Particle();
    Particle(double x, double y, double z, double sigma);

    __host__ __device__ double get_x() const;
    __host__ __device__ double get_y() const;
    __host__ __device__ double get_z() const;
    __host__ __device__ double get_sigma() const;
    __host__ __device__ size_t get_id() const;

    void set_x(double x);
    void set_y(double y);
    void set_z(double z);
    void set_sigma(double new_sigma);
};

#endif //PARTICLE_FILE_H
