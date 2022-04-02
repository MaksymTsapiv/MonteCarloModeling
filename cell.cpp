#include <algorithm>
#include "cell.h"

idx3d Cell::get_index() const {
    return index;
}

void Cell::add_particle(int id) {
    particlesId.push_back(id);
}

void Cell::remove_particle(int id) {
    for (auto i = 0; i < particlesId.size(); i++) {
        if (i == id) {
            particlesId.erase(particlesId.begin() + i);
            return;
        }
    }
}

std::vector<int> Cell::get_particles() const {
    return particlesId;
}
