#ifndef MODEL_GRID_CUH
#define MODEL_GRID_CUH

#include <map>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

#include "particle.cuh"
#include "d3.cuh"
#include "array.cuh"

class Grid {
private:
    /* Number of cells per each dimention */
    D3<uint> dim_cells {0};
    D3<double> cell_size {0.0};
    std::vector<Particle> particles{};

    /* Number of particles in each cell */
    uint *partPerCell;

    /* number of particle in system */
    size_t n = 0;

    /* number of cells in system */
    size_t n_cells = 0;

    /* Grid particle's sigma -- diameter */
    double p_sigma = 1.0;

    /************************ On GPU ************************/
    D3<double> *cudaL;
    OrderedArray particles_ordered;

    // Helper boolean array, needed in kernel funciton during intersection check
    int *intersectsCuda;

    uint *cellStartIdx;
    /********************************************************/

public:
    Grid(double x, double y, double z, D3<uint> dim_cells_, size_t n_particles) :
        particles_ordered(n_particles), n(n_particles), dim_cells{dim_cells_}
    {
        L = D3<double>{x, y, z};
        cell_size = D3<double>{L.x / dim_cells.x, L.y / dim_cells.y, L.z / dim_cells.z};

        if (cell_size.x < 1.0 || cell_size.y < 1.0 || cell_size.z < 1.0)
            throw std::runtime_error("Cell size is less than 1.0, e.g. smaller than particle size");

        n_cells = dim_cells.x * dim_cells.y * dim_cells.z;

        cudaMalloc(&cudaL, sizeof(D3<double>));
        cudaMemcpy(cudaL, &L, sizeof(D3<double>), cudaMemcpyHostToDevice);

        partPerCell = new uint[n_cells];
        memset(partPerCell, 0, sizeof(uint) * n_cells);

        cudaMalloc(&cellStartIdx, sizeof(uint) * n_cells);
        cudaMemset(cellStartIdx, 0, sizeof(uint) * n_cells);

        cudaMalloc(&intersectsCuda, sizeof(int));
        cudaMemset(intersectsCuda, 0, sizeof(int));

        print_grid_info();
    }
    ~Grid() {
        cudaFree(cudaL);
        cudaFree(intersectsCuda);
        cudaFree(cellStartIdx);
    }
    Grid operator=(const Grid &grid) = delete;

    D3<double> L {0.0};

    template<typename T>
    D3<T> normalize(D3<T> p) const;

    /* returns cell coordinates in 3D space -- (x,y,z) */
    template <typename T>
    D3<int> get_cell(D3<T> p) const;

    template <typename T>
    size_t cell_id(D3<T> p) const;


    std::vector<size_t>
    check_intersect_cpu(Particle particle);

    std::vector<size_t>
    check_intersect_cpu(Particle particle, uint cell_id);

    std::vector<size_t>
    check_intersect_cpu(Particle particle, uint cell_id, uint particle_id);


    template <typename T>
    uint cell_at_offset(D3<uint> init_cell, D3<T> offset) const;

    /*
     * Insert particle into <particles_ordered> array, updating <cellStartIdx> and <partPerCell>.
     *  It also adds particleto CPU <particles> vector
     *  Basically, it does everything needed to insert paticle and preserve correctness of grid.
     */
    void complex_insert(Particle p);


    void fill();
    void fill_cpu();

    void move(double dispmax);
    void export_to_pdb(const std::string& fn);

    /* Import and export to Custom Format (cf) */

    void export_to_cf(const std::string& fn);

    /*
     * import_from_cf expects that constructor has already been called, number of cells per
     * dimention and grid size are set
     */
    void import_from_cf(const std::string& fn);

    std::vector<Particle> get_particles() const;
    Particle get_particle(uint id) const;
    
    /* Actual number of particles in vector of particles */
    size_t de_facto_n() const;

    double volume() const;

    /* density() and packing_fraction() computes value for desired n, not actual */
    double density() const;
    double packing_fraction() const;

    void print_grid_info() const;
};


double random_double(double from, double to);

#endif //MODEL_GRID_CUH
