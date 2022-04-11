// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <cstddef>
#include "particle.cuh"

size_t Particle::nextId = 0;

Particle::Particle() : x_cor(0), y_cor(0), z_cor(0), sigma(0), id(nextId)
{ nextId++; }


Particle::Particle(double x, double y, double z, double sigma) :
    x_cor(x), y_cor(y), z_cor(z), sigma(sigma), id(nextId)
{ nextId++; }


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

size_t Particle::get_id() const{
    return id;
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
