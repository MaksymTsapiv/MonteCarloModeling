#ifndef QUAT_FILE_H
#define QUAT_FILE_H

#include <random>
#include <vector>
#include <Eigen/Dense>
#include "grid.cuh"

struct Quaternion {
    double a, b, c, d;
};

/* Generates random quaternion
 *  Uses formula 4.57 from book M. Allen, D. Tildesley "Computer Simulation of Liquids"
 */
Quaternion randomQuaternion();

Quaternion quatmul(const Quaternion& q1, const Quaternion& q2);

/* Transforms quaternion to rotation matrix
 *  Uses formula 3.40 from book M. Allen, D. Tildesley "Computer Simulation of Liquids"
 */
Eigen::Matrix<double, 3, 3> quatToRotMatrix(const Quaternion &q);

/*
 * Returns rotation about an axis defined by a unit vector <n_hat>
 *  (for 3D space it should have 3 components),
 *  through an angle <phi> (in radians). Rotation is represented by a quaternion
 *  Uses formula 3.38 from book M. Allen, D. Tildesley "Computer Simulation of Liquids"
 */
Quaternion singleRotQuaternion(double phi, const std::vector<double> &n_hat);

/*
 * Update coordinates after rotation
 *  Changes coordinates <coord> under rotation matrix <mat>
 */
void updCoordRot(D3<double> &coord, const std::vector<double> &mat);

/*
 * Generate a random unit vector
 */
D3<double> randVector();

Quaternion rotQuat(double angle, D3<double> axis, const Quaternion &old);

Quaternion randRotQuat(double angle_max, const Quaternion &old);

#endif //QUAT_FILE_H