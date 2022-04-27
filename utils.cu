#include <cmath>
#include "utils.cuh"
#include "particle.cuh"
#include "grid.cuh"

static unsigned int n4rdf(double r, double dr, Particle &p, Grid &grid) {
    unsigned int n_within_r = 0;
    unsigned int n_within_dr = 0;

    uint val = 0;

    // TODO: implement iterating only in cells that are at most <r> from particle
    for (auto particle : grid.get_particles()) {
        auto dist = grid.distance(particle.id, p.id);
        if (dist <= r+dr && dist > r)
            val++;
    }
    return val;
}

double rdf(double r, double dr, Grid &grid) {
    double density = grid.density();

    unsigned int n_sum = 0;
    for (auto particle : grid.get_particles()) {
        n_sum = n4rdf(r, dr, particle, grid);
    }

    double n = static_cast<double>(n_sum) / grid.n_particles();

    double rdf = static_cast<double>(3*n / ( 4*M_PI * ( pow(r+dr, 3)-pow(r, 3) ) * density));

    return rdf;
}
