#ifndef MODEL_GRID_H
#define MODEL_GRID_H

#include <vector>
#include "particle.h"


class Grid {
private:
    double Lx, Ly, Lz;
    std::vector<Particle> particles{};


public:
    Grid(double x, double y, double z) : Lx(x), Ly(y), Lz(z) {};

    void fill(size_t n);
    void move();
};

double random_double(double from, double to);

double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_H
