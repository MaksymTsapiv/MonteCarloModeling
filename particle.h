#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

class Particle {
private:
    double x_cor, y_cor, z_cor, sigma;
    unsigned int id;

public:
    static unsigned int nextId;

    Particle();
    Particle(double x, double y, double z, double sigma);

    double get_x() const;
    double get_y() const;
    double get_z() const;
    double get_sigma() const;
    double get_id() const;

    void set_x(double x);
    void set_y(double y);
    void set_z(double z);
    void set_sigma(double new_sigma);
};

#endif //PARTICLE_FILE_H
