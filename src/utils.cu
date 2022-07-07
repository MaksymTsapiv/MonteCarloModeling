#include <cmath>
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
    auto n_parts = grid.de_facto_n();

    auto grid_size = grid.L;

    for (int i = 0; i < n_parts-1; i++) {
        for (int j = i+1; j < n_parts; j++) {
            double r = pbs_distance(particles[i].get_coord(), particles[j].get_coord(), grid_size);
            int k = floor(r/dr);

            if (k >= nk)
                continue;

            rdf[k]++;
        }
    }

    auto density = grid.density();

    for (int k = 0; k < nk; k++) {
        auto rk = k*dr;
        rdf[k] = rdf[k] / ( 2.0/3.0 * M_PI * (pow(rk+dr, 3) - pow(rk, 3))) / n_parts / density;
    }

    return rdf;
}

std::vector<double> compute_rdf(const Grid &grid, double dr, double rmax,
                                const std::vector<double> &rdf_prev) {

    auto rdf_this = compute_rdf(grid, dr, rmax);

    if (rdf_this.size() != rdf_prev.size())
        throw std::runtime_error("Error, number of entries in previous vector of RDF values and\
                current vector of RDF values are not the same.");

    for (size_t i = 0; i < rdf_this.size(); i++)
        rdf_this[i] = (rdf_this[i]+rdf_prev[i])/2;

    return rdf_this;
}

void save_rdf_to_file(std::vector<double> rdf, double dr, double rmax, const std::string &fn) {

    std::ofstream RDFFile {fn, std::ofstream::out};

    if (!RDFFile.is_open())
        throw std::runtime_error("Error while opening RDFFile for energy " + fn);

    for (int k = 0; k < rdf.size(); k++) {
        auto kdrStr = std::to_string(k*dr);

        if (kdrStr.size() > DAT_FIRST_COL_LENGTH)
            std::cerr << "Warning: k*dr in casted to string is so big that it exceeds "
                "DAT_FIRST_ROW_LENGTH constexpr (space reserved for first column). "
                ".dat file with RDF may be corrupted now (consider increasing "
                "this constexpr, it is save to do so if you need)" << std::endl;

        for (auto i = kdrStr.size(); i < DAT_FIRST_COL_LENGTH; i++)
            kdrStr += " ";

        RDFFile << k*dr << " " << rdf[k] << std::endl;
    }

    RDFFile.close();
}
