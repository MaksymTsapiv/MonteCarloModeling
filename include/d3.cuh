#ifndef DIMENTION_3_CUH
#define DIMENTION_3_CUH

template <typename T>
class D3 {
public:
    T x, y, z;
    __host__ __device__ D3(T x, T y, T z) : x(x), y(y), z(z) {}
    D3(T val) : x(val), y(val), z(val) {}

    D3 operator+(const D3& other) const {
        return D3{ x+other.x, y+other.y, z+other.z };
    }
    D3<double> toD3double() const {
        return D3<double>{ static_cast<double>(x), static_cast<double>(y), static_cast<double>(z) };
    }
};
#endif //DIMENTION_3_CUH
