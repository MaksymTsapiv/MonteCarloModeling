#ifndef MODEL_GRID_H
#define MODEL_GRID_H

#include <map>
#include <vector>
#include <string>
#include "particle.h"
#include "cell.h"

class Grid {
private:
    int dim_cells = 10;
    double Lx{}, Ly{}, Lz{};
    std::vector<Particle> particles{};
    std::vector<Cell> cells{};
    std::map<size_t, std::vector<size_t>> compute_adj_cells() const;
    void common_initializer(double x, double y, double z);

public:
    Grid(double x, double y, double z, int dim_cells_);
    Grid(double x, double y, double z);
    Grid operator=(const Grid &grid) = delete;

    double get_Lx() const;
    double get_Ly() const;
    double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    size_t get_cell_id(double x, double y, double z) const;

    void fill(size_t n);
    void move(double dispmax);
    void export_to_pdb(std::string fn);

    Particle get_particle(size_t id);
};


Grid::Grid(double x, double y, double z) {
    common_initializer(x, y, z);
}

Grid::Grid(double x, double y, double z, int dim_cells_) {
    dim_cells = dim_cells_;
    common_initializer(x, y, z);
}


double random_double(double from, double to);

double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_H
