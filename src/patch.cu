#include "patch.cuh"

Patch::Patch() : x(0), y(0), z(0), sigma(0), type(0) {
}

Patch::Patch(double x_, double y_, double z_, double sigma_, size_t type_) :
        x(x_), y(y_), z(z_), sigma(sigma_), type(type_) {

}
