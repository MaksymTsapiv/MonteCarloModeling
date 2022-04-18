#ifndef MODEL_GRID_CUH
#define MODEL_GRID_CUH

#include <map>
#include <vector>
#include <string>
#include "particle.cuh"
#include "d3.cuh"
#include "array.cuh"

class Grid {
private:
    D3<double> dim_cells {10.0};
    D3<double> L {10.0};
    std::vector<Particle> particles{};
    OrderedArray particles_ordered;

    // On GPU
    D3<double> *cudaL;
    uint *cellStartIdx;
    uint *cellEndIdx;

public:
    Grid(double x, double y, double z, D3<double> dim_cells_) : Grid(x, y, z) {
        dim_cells = dim_cells_;
    }
    Grid(double x, double y, double z, double dim_cells_) : Grid(x, y, z) {
        dim_cells = D3{dim_cells_};
    }
    Grid(double x, double y, double z) : particles_ordered(255) {
        L = D3<double>{x, y, z};
        cudaMalloc(&cudaL, sizeof(D3<double>));
        cudaMemcpy(cudaL, &L, sizeof(D3<double>), cudaMemcpyHostToDevice);
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
