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
    int n_cells_dim = 0;
    double Lx, Ly, Lz;
    std::vector<Particle> particles{};
    std::vector<Cell> cells{};
    void common_initializer(int x, int y, int z);
    std::map<int, std::vector<int>> compute_adj_cells() const;

public:
    Grid() : Lx(0), Ly(0), Lz(0) {}
    Grid(double x, double y, double z, int dim_cells_);

    Grid operator=(const Grid &grid) = delete;

    double get_Lx() const;
    double get_Ly() const;
    double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    double get_density() const;
    double get_volume() const;
    size_t get_num_particles() const;
    std::vector<Particle> get_particles() const;

    double distance(int id1, int id2) const;

    int get_cell_id(int x, int y, int z) const;

    void fill(size_t n);
    void move(double dispmax);
    void export_to_pdb(std::string fn);

    Particle get_particle(int id) const;
};

double random_double(double from, double to);

#endif //MODEL_GRID_H
