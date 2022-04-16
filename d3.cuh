#ifndef DIMENTION_3_CUH
#define DIMENTION_3_CUH

template <typename T>
class D3 {
public:
    T x, y, z;
    D3(T x, T y, T z) : x(x), y(y), z(z) {}
    D3(T val) : x(val), y(val), z(val) {}
};
#endif //DIMENTION_3_CUH
