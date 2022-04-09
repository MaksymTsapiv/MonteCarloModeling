#include <cmath>
#include "utils.h"
#include "particle.h"
#include "grid.h"

double calc_dist(Particle p1, Particle p2) {
    double x1 = p1.get_x();
    double x2 = p2.get_x();
    double y1 = p1.get_y();
    double y2 = p2.get_y();
    double z1 = p1.get_z();
    double z2 = p2.get_z();

    // TODO: do benchmarking to find out time
    return hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
    // return sqrt(pow(sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2)), 2), pow((z1 -z2), 2));
}

static unsigned int n4rdf(double r, double dr, Particle &p, Grid &grid) {
    unsigned int n_within_r = 0;
    unsigned int n_within_dr = 0;

    // TODO: implement iterating only in cells that are at most <r> from particle
    for (auto particle : grid.get_particles()) {
        auto dist = grid.distance(particle.get_id(), p.get_id());
        if (dist <= r) {
            n_within_r++;
            n_within_dr++;
        }
        else if (dist <= r+dr)
            n_within_dr++;
    }
    return n_within_dr - n_within_r;
}

double rdf(double r, double dr, Grid &grid) {
    double density = grid.get_density();

    unsigned int n_sum = 0;
    for (auto particle : grid.get_particles()) {
        n_sum = n4rdf(r, dr, particle, grid);
    }

    double n = static_cast<double>(n_sum) / grid.get_num_particles();

    double rdf = static_cast<double>(3*n / ( 4*M_PI * ( pow(r+dr, 3)-pow(r, 3) ) * density));

    return rdf;
}
