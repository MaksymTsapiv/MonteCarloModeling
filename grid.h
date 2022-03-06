#ifndef MODEL_GRID_H
#define MODEL_GRID_H

#include <cstdio>
#include <vector>
#include "particle.h"


class Grid {
private:
    double Lx, Ly, Lz;
    std::vector<Particle> particles{};


public:
    void fill(size_t n);
    void move();
};

#endif //MODEL_GRID_H
