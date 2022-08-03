#ifndef MODEL_PATCH_CUH
#define MODEL_PATCH_CUH


class Patch {

public:
    double x{}, y{}, z{}, sigma{};
    size_t type{};

    Patch();
    Patch(double x_, double y_, double z_, double sigma_, size_t type_);
};


#endif //MODEL_PATCH_CUH
