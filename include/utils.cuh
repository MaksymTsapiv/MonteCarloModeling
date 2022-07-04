#ifndef UTILS_H
#define UTILS_H

#include "particle.cuh"
#include "grid.cuh"

std::vector<double> compute_rdf(const Grid &grid, double dr, double rmax);
/* Overload that accepts previous RDF and averages current RDF with previous RDF */
std::vector<double> compute_rdf(const Grid&, double, double, const std::vector<double>&);
void save_rdf_to_file(std::vector<double> rdf, double dr, double rmax, const std::string &fn);
double pbs_distance(double x1, double y1, double z1, double x2, double y2, double z2, D3<double>L);

#endif //UTILS_H
