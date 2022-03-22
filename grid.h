#ifndef MODEL_GRID_H
#define MODEL_GRID_H

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

    double get_Lx() const;
    double get_Ly() const;
    double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    void fill(size_t n);
    void move();
    void export_to_pdb(std::string fn);
};

double random_double(double from, double to);

double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_H
