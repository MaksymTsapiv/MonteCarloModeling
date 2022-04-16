#ifndef MODEL_GRID_CUH
#define MODEL_GRID_CUH

#include <map>
#include <vector>
#include <string>
#include "particle.cuh"
#include "cell.cuh"
#include "d3.cuh"

class Grid {
private:
    D3<double> dim_cells {10};
    double Lx{}, Ly{}, Lz{};
    std::vector<Particle> particles{};

    void common_initializer(double x, double y, double z);

public:
    Grid(double x, double y, double z, double dim_cells_) {
        dim_cells = D3{dim_cells_};
        common_initializer(x, y, z);
    }
    Grid(double x, double y, double z) {
        common_initializer(x, y, z);
    }
    Grid operator=(const Grid &grid) = delete;

    __host__ __device__ double get_Lx() const;
    __host__ __device__ double get_Ly() const;
    __host__ __device__ double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    size_t get_cell_id(double x, double y, double z) const;

    void fill(size_t n);
    void move(double dispmax);
    void export_to_pdb(std::string fn);

    __host__ __device__ Particle get_particle(size_t id);
};


double random_double(double from, double to);

__host__ __device__ double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_CUH
