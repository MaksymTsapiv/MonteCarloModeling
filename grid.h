#ifndef MODEL_GRID_H
#define MODEL_GRID_H

#include <cstdio>
#include <vector>
#include <string>
#include "particle.h"


class Grid {
private:
    double Lx, Ly, Lz;
    std::vector<Particle> particles{};

public:
    Grid() : Lx(0), Ly(0), Lz(0) {};
    Grid(double x, double y, double z) : Lx(x), Ly(y), Lz(z) {};

    double get_Lx();
    double get_Ly();
    double get_Lz();

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    void fill(size_t n);
    void move();
    void export_to_pdb(std::string fn);
};

#endif //MODEL_GRID_H
