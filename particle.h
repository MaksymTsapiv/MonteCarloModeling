#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

class Particle {
private:
    double x_cor, y_cor, z_cor, sigma;
    size_t id, cluster_id;

public:
    static size_t nextId;

    Particle();
    Particle(double x, double y, double z, double sigma);

    [[nodiscard]] double get_x() const;
    [[nodiscard]] double get_y() const;
    [[nodiscard]] double get_z() const;
    [[nodiscard]] double get_sigma() const;
    [[nodiscard]] size_t get_id() const;
    [[nodiscard]] size_t get_cluster_id() const ;

    void set_x(double x);
    void set_y(double y);
    void set_z(double z);
    void set_sigma(double new_sigma);
    void set_cluster_id(size_t cluster);
};

#endif //PARTICLE_FILE_H
