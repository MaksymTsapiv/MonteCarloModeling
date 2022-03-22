#ifndef PARTICLE_FILE_H
#define PARTICLE_FILE_H

class Particle {
private:
    double x_cor, y_cor, z_cor, sigma;

public:
    Particle() : x_cor(0), y_cor(0), z_cor(0), sigma(0) {};
    Particle(double x, double y, double z, double sigma) : x_cor(x), y_cor(y), z_cor(z), sigma(sigma) {};

    double get_x() const;
    double get_y() const;
    double get_z() const;
    double get_sigma() const;

    void set_x(double x);
    void set_y(double y);
    void set_z(double z);
    void set_sigma(double new_sigma);
};


#endif //PARTICLE_FILE_H
