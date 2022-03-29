#ifndef MODEL_GRID_H
#define MODEL_GRID_H

#include <vector>
#include <string>
#include "particle.h"
#include "cell.h"

class Grid {
private:
    unsigned int dim_cells = 100;
    double Lx, Ly, Lz;
    std::vector<Particle> particles{};
    std::vector<Cell> cells{};

public:
    Grid() : Lx(0), Ly(0), Lz(0) {}
    Grid(double x, double y, double z, unsigned int dim_cells_) : Lx(x), Ly(y), Lz(z), dim_cells(dim_cells_) {
        cells.reserve(dim_cells_ * dim_cells_ * dim_cells_);
    }
    Grid(double x, double y, double z) {
        Grid(x, y, z, dim_cells);
    }
    Grid operator=(const Grid &grid) = delete;

    double get_Lx() const;
    double get_Ly() const;
    double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    int get_cell_id(unsigned int x, unsigned int y, unsigned int z) const;

    void fill(size_t n);
    void move();
    void export_to_pdb(std::string fn);
};

double random_double(double from, double to);

double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_H
