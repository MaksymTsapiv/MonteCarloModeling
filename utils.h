#ifndef UTILS_H
#define UTILS_H

#include "particle.h"
#include "grid.h"

double rdf(double r, double dr, Grid &grid);
double calc_dist(Particle p1, Particle p2);
double calc_dist(Particle p1, Particle p2, double Lx, double Ly, double Lz);

#endif //UTILS_H
