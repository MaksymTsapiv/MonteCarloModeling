#ifndef UTILS_H
#define UTILS_H

#include "particle.cuh"
#include "grid.cuh"

std::vector<double> compute_rdf(const Grid &grid, double dr, double rmax);
/* Overload that accepts previous RDF and averages current RDF with previous RDF */
std::vector<double> compute_rdf(const Grid&, double, double, const std::vector<double>&);
void save_rdf_to_file(std::vector<double> rdf, double dr, double rmax, const std::string &fn);
double pbs_distance(D3<double> p1, D3<double> p2, D3<double> L);

/* Helping function to print distances between each particle's center and its patches */
void checkPatchDist(const Grid &grid);

#endif //UTILS_H
