#ifndef MODEL_CELL_CUH
#define MODEL_CELL_CUH

#include <vector>
#include <string>
#include "particle.cuh"

struct idx3d {
    int x, y, z;
};

class Cell {
private:
    const idx3d index;
    std::vector<size_t> particlesId;

public:
    Cell() = delete;
    Cell(int x, int y, int z) : index{x, y, z} {}

    idx3d get_index() const;
    void add_particle(size_t id);
    void remove_particle(size_t id);
    std::vector<size_t> get_particles() const;
};

#endif //MODEL_CELL_CUH
