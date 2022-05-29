#ifndef MODEL_GRID_CUH
#define MODEL_GRID_CUH

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <random>

#include "particle.cuh"
#include "d3.cuh"
#include "array.cuh"

constexpr double SPHERE_PACK_COEFF = 0.9069;
constexpr uint nAdjCells = 27;    // Number of neighbouring cells

struct AdjCells { 
    uint ac[nAdjCells];    // id of cell
};


class Grid {
private:
    /* Number of cells per each dimention */
    D3<uint> dimCells {0};
    D3<double> cellSize {0.0};
    std::vector<Particle> particles{};

    /* Number of particles in each cell */
    uint *partPerCell;

    /* number of particle in system */
    size_t n = 0;

    /* number of cells in system */
    size_t n_cells = 0;
    uint maxPartPerCell = 0;
    uint maxPartPerCell2pow = 0;
    double energy = 0;

    /* Grid particle's sigma -- diameter */
    const double pSigma = 1.0;
    const double temp = 0.5;
    const double beta = 1.0/temp;

    AdjCells *cn;

    /************************ On GPU ************************/

    AdjCells *cnCuda;
    uint *partPerCellCuda;

    D3<double> *cudaL;
    OrderedArray orderedParticlesCuda;

    /* Helper array of 27 doubles that stores cells energy for all adjacent cells */
    double *energiesCuda;

    /* Helper boolean array, needed in kernel funciton during intersection check */
    int *intersectsCuda;

    uint *cellStartIdxCuda;
    /********************************************************/

public:
    Grid(double x, double y, double z, D3<uint> dim_cells_, size_t n_particles) :
        orderedParticlesCuda(n_particles), n(n_particles), dimCells{dim_cells_}
    {
        L = D3<double>{x, y, z};
        cellSize = D3<double>{L.x / dimCells.x, L.y / dimCells.y, L.z / dimCells.z};

        if (cellSize.x < 1.0 || cellSize.y < 1.0 || cellSize.z < 1.0)
            throw std::runtime_error("Cell size is less than 1.0, e.g. smaller than particle size");

        n_cells = dimCells.x * dimCells.y * dimCells.z;

        cudaMalloc(&cudaL, sizeof(D3<double>));
        cudaMemcpy(cudaL, &L, sizeof(D3<double>), cudaMemcpyHostToDevice);

        partPerCell = new uint[n_cells];
        memset(partPerCell, 0, sizeof(uint) * n_cells);

        maxPartPerCell = SPHERE_PACK_COEFF * (3.0 * cellSize.x*cellSize.y*cellSize.z)/
                                    (4.0 * M_PI * pow(static_cast<double>(pSigma)/2.0, 3));

        maxPartPerCell2pow = pow(2, ceil(log2(maxPartPerCell)));

        cudaMalloc(&cellStartIdxCuda, sizeof(uint) * n_cells);
        cudaMemset(cellStartIdxCuda, 0, sizeof(uint) * n_cells);

        cudaMalloc(&intersectsCuda, sizeof(int));
        cudaMemset(intersectsCuda, 0, sizeof(int));

        cudaMalloc(&energiesCuda, nAdjCells*sizeof(double));
        cudaMemset(energiesCuda, 0, nAdjCells*sizeof(double));

        cn = (AdjCells*) malloc(n_cells*sizeof(AdjCells));
        compute_adj_cells();

        cudaMalloc(&cnCuda, n_cells*sizeof(AdjCells));
        cudaMemcpy(cnCuda, cn, n_cells*sizeof(AdjCells), cudaMemcpyHostToDevice);

        cudaMalloc(&partPerCellCuda, n_cells*sizeof(uint));

        print_grid_info();
    }
    ~Grid() {
        cudaFree(cudaL);
        cudaFree(intersectsCuda);
        cudaFree(cellStartIdxCuda);
        free(cn);
    }
    Grid operator=(const Grid &grid) = delete;

    D3<double> L {0.0};

    template<typename T>
    D3<T> normalize(D3<T> p) const;

    /* returns cell coordinates in 3D space -- (x,y,z) */
    template <typename T>
    D3<int> cell(D3<T> p) const;

    template <typename T>
    size_t cell_id(D3<T> p) const;

    double get_energy() const {
        return energy;
    }

    std::vector<size_t>
    check_intersect_cpu(Particle particle);

    std::vector<size_t>
    check_intersect_cpu(Particle particle, uint cell_id);

    std::vector<size_t>
    check_intersect_cpu(Particle particle, uint cell_id, uint particle_id);

    /*
     * Insert particle into <particles_ordered> array, updating <cellStartIdx> and <partPerCell>.
     *  It also adds particleto CPU <particles> vector
     *  Basically, it does everything needed to insert paticle and preserve correctness of grid.
     */
    void complex_insert(Particle p);


    void fill();
    size_t move(double dispmax);
    void export_to_pdb(const std::string& fn);

    /* Import and export to Custom Format (cf) */

    void export_to_cf(const std::string& fn);

    /*
     * import_from_cf expects that constructor has already been called, number of cells per
     * dimention and grid size are set
     */
    void import_from_cf(const std::string& fn);

    void system_energy();

    std::vector<Particle> get_particles() const;
    Particle get_particle(uint id) const;
    
    /* Actual number of particles in vector of particles */
    size_t de_facto_n() const;

    double volume() const;

    /* density() and packing_fraction() computes value for desired n, not actual */
    double density() const;
    double packing_fraction() const;

    void compute_adj_cells();

    void print_grid_info() const;
};


double random_double(double from, double to);
int random_int(int from, int to);

#endif //MODEL_GRID_CUH
