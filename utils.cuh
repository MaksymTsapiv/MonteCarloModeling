#ifndef UTILS_H
#define UTILS_H

#include "particle.cuh"
#include "grid.cuh"

std::vector<double> compute_rdf(const Grid &grid, double dr, double rmax);
void save_rdf_to_file(std::vector<double> rdf, double dr, double rmax, std::string filename);
double pbs_distance(double x1, double y1, double z1, double x2, double y2, double z2, D3<double>L);

#endif //UTILS_H
