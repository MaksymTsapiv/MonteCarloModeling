// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "particle.h"

double Particle::get_x() const{
    return x_cor;
}

double Particle::get_y() const{
    return y_cor;
}

double Particle::get_z() const{
    return z_cor;
}

double Particle::get_sigma() const{
    return sigma;
}

void Particle::set_x(double x) {
    x_cor = x;
}

void Particle::set_y(double y) {
    y_cor = y;
}

void Particle::set_z(double z) {
    z_cor = z;
}

void Particle::set_sigma(double new_sigma) {
    sigma = new_sigma;
}
