#include <algorithm>
#include "cell.h"

idx3d Cell::get_index() const {
    return index;
}

void Cell::add_particle(size_t id) {
    particlesId.push_back(id);
}

void Cell::remove_particle(size_t id) {
    for (size_t i = 0; i < particlesId.size(); ++i) {
        if (i == id) {
            particlesId.erase(particlesId.begin() + i);
            return;
        }
    }
}

std::vector<size_t> Cell::get_particles() const {
    return particlesId;
}
