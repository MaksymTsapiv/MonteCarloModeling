// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <cstddef>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include "grid.h"
#include "particle.h"


double random_double(double from, double to) {
    std::random_device rd;
    std::mt19937 rand_double(rd());

    std::uniform_real_distribution<> dist(from, to);
    return dist(rand_double);
}


double calc_dist(Particle p1, Particle p2) {
    double x1 = p1.get_x();
    double x2 = p2.get_x();
    double y1 = p1.get_y();
    double y2 = p2.get_y();
    double z1 = p1.get_z();
    double z2 = p2.get_z();

    return hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
}


void Grid::fill(size_t n) {
    bool flag = true;
    size_t count_tries = 0;
    size_t max_tries = 10000 + n;

    double sigma = 1.0;

    double x = (Lx - sigma) * random_double(0, 1);
    double y = (Ly - sigma) * random_double(0, 1);
    double z = (Lz - sigma) * random_double(0, 1);

    Particle particle = Particle(x, y, z, sigma);
    particles.push_back(particle);

    while ((particles.size() != n) && count_tries != max_tries) {

        x = (Lx - sigma) * random_double(0, 1);
        y = (Ly - sigma) * random_double(0, 1);
        z = (Lz - sigma) * random_double(0, 1);

        particle = Particle(x, y, z, sigma);

        for (Particle p : particles) {
            if (calc_dist(p, particle) < sigma) {  //TODO: >= or > ???
                flag = false;
                break;
            }
        }

        if (flag) particles.push_back(particle);
        flag = true;

        count_tries++;
    }

    std::cout << count_tries << "\n" <<particles.size() << std::endl;

}

void Grid::move() {

}
