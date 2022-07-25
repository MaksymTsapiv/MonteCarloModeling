#include <vector>
#include <Eigen/Dense>
#include "quat.cuh"
#include "d3.cuh"

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
        s2 = pow(z3, 2) + pow(z4, 2);
    }
    while (s2 >= 1);


    double sqr_eq = sqrt((1-s1)/s2);
    Quaternion a {z1, z2, z3*sqr_eq, z4*sqr_eq};

    return a;
}

Eigen::Matrix<double, 3, 3> quatToRotMatrix(const Quaternion &q) {
    /* Construct vector of 9 elements
     *  (3*3 = 9 because we reprecent 3x3 matrix as linear vector for simplicity)
     */
    Eigen::Matrix<double, 3, 3> rotMat{
        {
            pow(q.a, 2) + pow(q.b, 2) - pow(q.c, 2) - pow(q.d, 2),
            2 * (q.b*q.c + q.a*q.d),
            2 * (q.b*q.d - q.a*q.c),
        },
        {
            2 * (q.b*q.c - q.a*q.d),
            pow(q.a, 2) - pow(q.b, 2) + pow(q.c, 2) - pow(q.d, 2),
            2 * (q.c*q.d + q.a*q.b),
        },
        {
            2 * (q.b*q.d + q.a*q.c),
            2 * (q.c*q.d - q.a*q.b),
            pow(q.a, 2) - pow(q.b, 2) - pow(q.c, 2) + pow(q.d, 2),
        },
    };

    return rotMat;
}

Quaternion singleRotQuaternion(double phi, const std::vector<double> &n_hat) {
    double sin = std::sin(0.5*phi);
    Quaternion q{};
    q.a = std::cos(0.5*phi);
    q.b = n_hat[0] * sin;
    q.c = n_hat[1] * sin;
    q.d = n_hat[2] * sin;

    return q;
}

Quaternion quatmul(const Quaternion& q1, const Quaternion& q2) {
    return Quaternion{q1.a * q2.a - q1.b * q2.b - q1.c * q2.c - q1.d * q2.d,
                      q1.b * q2.a + q1.a * q2.b - q1.d * q1.c + q1.c * q2.d,
                      q1.c * q2.a + q1.d * q2.b + q1.a * q2.c - q1.b * q2.d,
                      q1.d * q2.a - q1.c * q2.b + q1.b * q2.c + q1.a * q2.d
    };
}


void updCoordRot(D3<double> &coord, const std::vector<double> &mat) {
    auto newX = coord.x*mat[0] + coord.x*mat[1] + coord.x*mat[2];
    auto newY = coord.y*mat[3] + coord.y*mat[4] + coord.y*mat[5];
    auto newZ = coord.z*mat[6] + coord.z*mat[7] + coord.z*mat[8];

    coord.x = newX;
    coord.y = newY;
    coord.z = newZ;
}

D3<double> randVector() {
    double x = random_double(-1, 1);
    double y = random_double(-1, 1);
    double z = random_double(-1, 1);

    double vecLen = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));

    D3<double> vec{x/vecLen, y/vecLen, z/vecLen};

    return vec;
}

Quaternion rotQuat(double angle, D3<double> axis, const Quaternion &old) {
    D3<double> rotVec { std::sin(0.5*angle) * axis.x,
                        std::sin(0.5*angle) * axis.y,
                        std::sin(0.5*angle) * axis.z};

    Quaternion rotQuat {std::cos(0.5*angle), rotVec.x, rotVec.y, rotVec.z};

    return quatmul(rotQuat, old);
}

Quaternion randRotQuat(double angle_max, const Quaternion &old) {
    D3<double> axis = randVector();
    double angle = ( 2.0*random_double(0, 1) - 1.0 ) * angle_max;
    Quaternion e = rotQuat(angle, axis, old);
    return e;
}
