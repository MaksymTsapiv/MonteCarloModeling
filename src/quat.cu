#include <vector>
#include "quat.cuh"

Quaternion randomQuaternion() {
    double s1, s2;
    double z1, z2, z3, z4;

    do {
        z1 = random_double(-1, 1);
        z2 = random_double(-1, 1);
        s1 = pow(z1, 2) + pow(z2, 2);
    }
    while (s1 >= 1);

    do {
        z3 = random_double(-1, 1);
        z4 = random_double(-1, 1);
        s1 = pow(z3, 2) + pow(z4, 2);
    }
    while (s2 >= 1);


    double sqr_eq = sqrt((1-s1)/s2);
    Quaternion a {z1, z2, z3*sqr_eq, z4*sqr_eq};

    return a;
}

std::vector<double> quatToRotMatrix(const Quaternion &q) {
    /* Construct vector of 9 elements
     *  (3*3 = 9 because we reprecent 3x3 matrix as linear vector for simplicity)
     */
    std::vector<double> rotMat(3*3, 0);

    rotMat[0] = pow(q.a, 2) + pow(q.b, 2) - pow(q.c, 2) - pow(q.d, 2);
    rotMat[1] = 2 * (q.b*q.c + q.a*q.d);
    rotMat[2] = 2 * (q.b*q.d - q.a*q.c);
    rotMat[3] = 2 * (q.b*q.c - q.a*q.d);
    rotMat[4] = pow(q.a, 2) - pow(q.b, 2) + pow(q.c, 2) - pow(q.d, 2);
    rotMat[5] = 2 * (q.c*q.d + q.a*q.b);
    rotMat[6] = 2 * (q.b*q.d + q.a*q.c);
    rotMat[7] = 2 * (q.c*q.d - q.a*q.b);
    rotMat[8] = pow(q.a, 2) - pow(q.b, 2) - pow(q.c, 2) + pow(q.d, 2);

    return rotMat;
}

Quaternion singleRotQuaternion(double phi, const std::vector<double> &n_hat) {
    double sin = std::sin(0.5*phi);
    Quaternion q;
    q.a = std::cos(0.5*phi);
    q.b = n_hat[0] * sin;
    q.c = n_hat[1] * sin;
    q.d = n_hat[2] * sin;

    return q;
}

void updCoordRot(D3<double> &coord, const std::vector<double> &mat) {
    auto newX = coord.x*matrix[0] + coord.x*matrix[1] + coord.x*matrix[2];
    auto newY = coord.y*matrix[3] + coord.y*matrix[4] + coord.y*matrix[5];
    auto newZ = coord.z*matrix[6] + coord.z*matrix[7] + coord.z*matrix[8];

    coord.x = newX;
    coord.y = newY;
    coord.z = newZ;
}
