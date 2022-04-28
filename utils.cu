#include <cmath>
#include <iostream>
#include <fstream>
#include "utils.cuh"
#include "particle.cuh"
#include "grid.cuh"

/* Periodic-Boundary safe distance (accounts for periodic boundary conditions) */
double pbs_distance(D3<double> p1, D3<double> p2, D3<double> L) {

    auto xd = std::min( fabs(p1.x - p2.x), L.x - fabs(p1.x - p2.x) );
    auto yd = std::min( fabs(p1.y - p2.y), L.y - fabs(p1.y - p2.y) );
    auto zd = std::min( fabs(p1.z - p2.z), L.z - fabs(p1.z - p2.z) );

    return hypot(hypot(xd, yd), zd);
}

std::vector<double> compute_rdf(const Grid &grid, double dr, double rmax) {

    int nk = static_cast<int>(rmax/dr) + 1;
    std::vector<double> rdf(nk, 0);

    std::vector<Particle> particles = grid.get_particles();
    auto n_parts = grid.n_particles();

    auto grid_size = grid.get_L();

    for (int i = 0; i < n_parts-1; i++) {
        for (int j = i+1; j < n_parts; j++) {
            double r = pbs_distance(particles[i].get_coord(), particles[j].get_coord(), grid_size);
            int k = floor(r/dr);

            if (k >= nk-1)
                continue;

            rdf[k]++;
        }
    }

    auto density = grid.density();

    std::cout << "Density: " << density << std::endl;

    for (int k = 0; k < nk; k++) {
        auto rk = k*dr;
        rdf[k] = rdf[k] / ( 4.0/3.0 * M_PI * (pow(rk+dr, 3) - pow(rk, 3)) * density);
    }

    return rdf;
}

void save_rdf_to_file(std::vector<double> rdf, double dr, double rmax, std::string filename) {

        std::ofstream file;
        file.open(filename);

        for (int k = 0; k < rdf.size(); k++)
            file << k*dr << " : " << rdf[k] << std::endl;

        file.close();
}
