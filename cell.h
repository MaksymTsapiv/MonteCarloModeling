#ifndef MODEL_CELL_H
#define MODEL_CELL_H

#include <vector>
#include <string>
#include "particle.h"

struct idx3d {
    int x, y, z;
};

class Cell {
private:
    const idx3d index;
    std::vector<int> particlesId;

public:
    Cell() = delete;
    Cell(int x, int y, int z) : index{x, y, z} {}

    idx3d get_index() const;
    void add_particle(int id);
    void remove_particle(int id);
};

#endif //MODEL_CELL_H
