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
    /* Number of cells per each dimention */
    D3<uint> dim_cells {10};
    D3<double> cell_size {0.0};
    D3<double> L {10.0};
    std::vector<Particle> particles{};

    /* number of particle in system */
    size_t n = 0;

    /* number of cells in system */
    size_t n_cells = 0;

    /************************ On GPU ************************/
    D3<double> *cudaL;
    OrderedArray particles_ordered;

    // Helper boolean array, needed in kernel funciton during intersection check
    bool *intersectsCuda;

    uint *cellStartIdx;
    /********************************************************/

public:
    Grid(double x, double y, double z, D3<uint> dim_cells_, size_t n_particles) :
        Grid(x, y, z, n_particles)
    {
        dim_cells = dim_cells_;
    }
    Grid(double x, double y, double z, size_t n_particles) : particles_ordered(n_particles), n(n_particles)
    {
        L = D3<double>{x, y, z};
        cell_size = D3 {L.x / dim_cells.x, L.y / dim_cells.y, L.z / dim_cells.z};

        n_cells = dim_cells.x * dim_cells.y * dim_cells.z;

        cudaMalloc(&cudaL, sizeof(D3<double>));
        cudaMemcpy(cudaL, &L, sizeof(D3<double>), cudaMemcpyHostToDevice);

        cudaMalloc(&cellStartIdx, sizeof(uint) * n_cells);
        cudaMemset(cellStartIdx, 0, sizeof(uint) * n_cells);

        cudaMalloc(&intersectsCuda, n_particles*sizeof(bool));
        cudaMemset(intersectsCuda, false, n_particles*sizeof(bool));
    }
    ~Grid() {
        cudaFree(cudaL);
        cudaFree(intersectsCuda);
        cudaFree(cellStartIdx);
    }
    Grid operator=(const Grid &grid) = delete;

    __host__ __device__ double get_Lx() const;
    __host__ __device__ double get_Ly() const;
    __host__ __device__ double get_Lz() const;

    void set_Lx(double x);
    void set_Ly(double y);
    void set_Lz(double z);

    template<typename T>
    D3<T> normalize(D3<T> p) const;

    /* returns cell coordinates in 3D space -- (x,y,z) */
    template <typename T>
    D3<uint> get_cell(D3<T> p) const;

    template <typename T>
    size_t cell_id(D3<T> p) const;

    template <typename T>
    uint cell_at_offset(D3<uint> init_cell, D3<T> offset) const;

    void fill();
    void move(double dispmax);
    void export_to_pdb(std::string fn);

    std::vector<Particle> get_particles() const;
    double distance(int id1, int id2) const;
    Particle get_particle(uint id) const;
    double density() const;
    size_t n_particles() const;
    double volume() const;
};


double random_double(double from, double to);

__host__ __device__ double calc_dist(Particle p1, Particle p2);

#endif //MODEL_GRID_CUH
