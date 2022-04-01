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
    double Lx, Ly, Lz;
    std::vector<Particle> particles{};
    std::vector<Cell> cells{};
    void common_initializer(int x, int y, int z);
    std::map<int, std::vector<int>> compute_adj_cells();

public:
    Grid() : Lx(0), Ly(0), Lz(0) {}
    Grid(double x, double y, double z, int dim_cells_);
    Grid(double x, double y, double z);

    Grid operator=(const Grid &grid) = delete;

    double get_Lx() const;
    double get_Ly() const;
    double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    int get_cell_id(int x, int y, int z) const;

    void fill(size_t n);
    void move(double dispmax);
    void export_to_pdb(std::string fn);
};

double random_double(double from, double to);

double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_H
