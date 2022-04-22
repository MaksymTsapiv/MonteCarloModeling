// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include "grid.cuh"
#include "particle.cuh"

__host__ __device__ double Grid::get_Lx() const{
    return L.x;
}
__host__ __device__ double Grid::get_Ly() const{
    return L.y;
}
__host__ __device__ double Grid::get_Lz() const{
    return L.z;
}

void Grid::set_Lx(double x) {
    L.x = x;
}
void Grid::set_Ly(double y) {
    L.y = y;
}
void Grid::set_Lz(double z) {
    L.z = z;
}

double random_double(double from, double to) {
    //std::random_device rd;
    //static std::mt19937 rand_double(rd());

    static std::mt19937 rand_double(1);

    std::uniform_real_distribution<> dist(from, to);
    return dist(rand_double);
}


__host__ __device__ double calc_dist(Particle p1, Particle p2) {
    double x1 = p1.x;
    double x2 = p2.x;
    double y1 = p1.y;
    double y2 = p2.y;
    double z1 = p1.z;
    double z2 = p2.z;

    return hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
    // return sqrt(pow(sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2)), 2), pow((z1 -z2), 2));
}

template <typename T>
D3<T> Grid::normalize(const D3<T> p) const {
    D3<double> new_p = p;

    if (p.x < 0)
        new_p.x = p.x + L.x;
    if (p.y < 0)
        new_p.y = p.y + L.y;
    if (p.z < 0)
        new_p.z = p.z + L.z;
    if (p.x >= L.x)
        new_p.x = p.x - L.x;
    if (p.y >= L.y)
        new_p.y = p.y - L.y;
    if (p.z >= L.z)
        new_p.z = p.z - L.z;

    return new_p;
}

template <typename T>
D3<uint> Grid::get_cell(D3<T> p) const {
    D3<double> new_p = normalize<double>(p.toD3double());

    uint c_x = static_cast<size_t>(floor( (new_p.x / L.x) * dim_cells.x) );
    uint c_y = static_cast<size_t>(floor( (new_p.y / L.y) * dim_cells.y) );
    uint c_z = static_cast<size_t>(floor( (new_p.z / L.z) * dim_cells.z) );
    D3<uint> cell{c_x, c_y, c_z};
    return cell;
}

template <typename T>
size_t Grid::cell_id(D3<T> p) const {
    D3<uint> cell = get_cell(p);
    return cell.x + cell.y * dim_cells.y + cell.z * dim_cells.z * dim_cells.z;
}

__device__ double device_min(double a, double b) {
    return a < b ? a : b;
}

__global__ void
check_intersect (
        const Particle *particle,
        const Particle *ordered_particles,
        uint *cellStartIdx,
        uint curr_cell_id,
        const D3<double> *L,
        bool *intersects) {

    uint startIdx = cellStartIdx[curr_cell_id];
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto xd = device_min( fabs(particle->x - ordered_particles[startIdx+idx].x),
                        L->x - fabs(particle->x - ordered_particles[startIdx+idx].x) );

    auto yd = device_min( fabs(particle->y - ordered_particles[startIdx+idx].y),
                        L->y - fabs(particle->y - ordered_particles[startIdx+idx].y) );

    auto zd = device_min( fabs(particle->z - ordered_particles[startIdx+idx].z),
                        L->z - fabs(particle->z - ordered_particles[startIdx+idx].z) );

    auto dist = hypot(hypot(xd, yd), zd);
    if (dist < particle->sigma)
        intersects[idx] = true;
    else
        intersects[idx] = false;
}

__global__ void update_kernel(uint *cellStartIdx, size_t cell_idx) {
    cellStartIdx[cell_idx+threadIdx.x]++;
}

void Grid::fill() {
    size_t count_tries = 0;
    size_t max_tries = 10000 + n;

    double sigma = 1.0;

    uint fails = 0;

    while ((particles.size() < n) && count_tries < max_tries) {

        double x = L.x * random_double(0, 1);
        double y = L.y * random_double(0, 1);
        double z = L.z * random_double(0, 1);

        Particle particle = Particle(x, y, z, sigma);

        Particle *cuda_particle;
        cudaMalloc(&cuda_particle, sizeof(Particle));
        cudaMemcpy(cuda_particle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);

        D3<double> p_point = particle.get_coord();
        D3<uint> p_cell = get_cell(p_point);
        uint particle_cell_id = cell_id(p_point);

        bool intersected = false;
        // TODO: Is it alright and accounts everything, using periodic boundary conditions?
        // TODO: reimplement loop using a more ellegant approach, because this is ugly
        for (double z_off = -cell_size.z; z_off <= cell_size.z; z_off+=cell_size.z) {
            for (double y_off = -cell_size.y; y_off <= cell_size.y; y_off+=cell_size.y) {
                for (double x_off = -cell_size.x; x_off <= cell_size.x; x_off+=cell_size.x) {
                    D3<double> offset = {x_off, y_off, z_off};
                    D3<uint> curr_cell = get_cell(p_point + offset);
                    uint curr_cell_id = cell_id(curr_cell);

                    // number of particles in cell
                    size_t partInCell = 0;
                    
                    // start index of cell in ordered array of particles
                    uint *csi_cci = new uint;
                    cudaMemcpy(csi_cci, (cellStartIdx+curr_cell_id), sizeof(uint),
                                                            cudaMemcpyDeviceToHost);

                    if (curr_cell_id == n_cells-1)
                        partInCell = particles.size() - *csi_cci;
                    else {
                        uint *csi_ccip = new uint;
                        cudaMemcpy(csi_ccip, (cellStartIdx+curr_cell_id+1), sizeof(uint),
                                                                    cudaMemcpyDeviceToHost);
                        partInCell = *csi_ccip - *csi_cci;
                    }

                    if (partInCell == 0)
                        continue;

                    const Particle *cuda_ordered_particles = particles_ordered.get_array();
                    // TODO: Variable block size
                    check_intersect<<<1, partInCell>>>( cuda_particle, cuda_ordered_particles,
                                                cellStartIdx, curr_cell_id, cudaL, intersectsCuda );

                    bool *intersects = new bool[partInCell];
                    cudaMemcpy(intersects, intersectsCuda, partInCell*sizeof(bool),
                                                            cudaMemcpyDeviceToHost);

                    for (auto i = 0; i < partInCell; i++) {
                        if (intersects[i]) {
                            intersected = true;
                            break;
                        }
                    }
                    delete[] intersects;

                }
                if (intersected) break;
            }
            if (intersected) break;
        }

        if (intersected) {
            fails++;
            continue;
        }

        if (!intersected) {
            particles.push_back(particle);
            auto cell_idx = cell_id(p_cell);

            // Cell start index of particle's cell
            uint *partCellStartIdx = new uint;
            cudaMemcpy(partCellStartIdx, &cellStartIdx[cell_idx], sizeof(uint),
                                                        cudaMemcpyDeviceToHost);

            particles_ordered.insert(particle, *partCellStartIdx);

            // TODO: Variable block size
            if (static_cast<int>(n_cells-cell_idx-1) > 0)
                update_kernel<<<1, n_cells-cell_idx-1>>>(cellStartIdx, cell_idx+1);
        }

        count_tries++;
        cudaFree(cuda_particle);
    }
    std::cout << "Fill complited with " << fails << " fails" << std::endl;
}

// This is temporary function just to make it work. TODO: make up a better design
__device__ double calc_dist(double x1, double y1, double z1, double x2, double y2, double z2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

__global__ void parent_kernel(Particle *p1, Particle *p, D3<double> grid_size, bool *intersects) {
    if (p1->id == p[threadIdx.x].id) {
        intersects[threadIdx.x] = false;
        return;
    }

    auto sigma = p1->sigma;

    auto x1 = p1->x;
    auto y1 = p1->y;
    auto z1 = p1->z;

    auto x2 = p[threadIdx.x].x;
    auto y2 = p[threadIdx.x].y;
    auto z2 = p[threadIdx.x].z;

    if (x1 >= grid_size.x/2)
        x1 -= grid_size.x;
    if (x2 >= grid_size.x/2)
        x2 -= grid_size.x;

    if (y1 >= grid_size.y/2)
        y1 -= grid_size.y;
    if (y2 >= grid_size.y/2)
        y2 -= grid_size.y;

    if (z1 >= grid_size.z/2)
        z1 -= grid_size.z;
    if (z2 >= grid_size.z/2)
        z2 -= grid_size.z;

    if (calc_dist(x1, y1, z1, x2, y2, z2) <= sigma) {
        intersects[threadIdx.x] = true;
        return;
    }
    intersects[threadIdx.x] = false;
}

void Grid::move(double dispmax) {
    size_t count = 0;
    double sigma = 1.0;

    for (auto & particle : particles) {
        double new_x = particle.x + random_double(-1, 1);
        double new_y = particle.y + random_double(-1, 1);
        double new_z = particle.z + random_double(-1, 1);

        double vec_x = new_x - particle.x;
        double vec_y = new_y - particle.y;
        double vec_z = new_z - particle.z;

        double vec_length = sqrt(pow(vec_x, 2) + pow(vec_y, 2) + pow(vec_z, 2));

        vec_x = vec_x / vec_length;
        vec_y = vec_y / vec_length;
        vec_z = vec_z / vec_length;

        double x = particle.x + vec_x * dispmax;
        double y = particle.y + vec_y * dispmax;
        double z = particle.z + vec_z * dispmax;

        if (x >= L.x) x -= L.x;
        if (y >= L.y) y -= L.y;
        if (z >= L.z) z -= L.z;

        if (x < 0) x += L.x;
        if (y < 0) y += L.y;
        if (z < 0) z += L.z;

        // TODO: implement
        // PROBLEM: with this approach we will have to iterate through the array of bools to
        //    check if any thread of kernel function returned true
        // Calculate <new_particle_coord_cell_id> -- cell id of the new particle position
        for (int z = -1; z <= 1; z++)
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++) {
                    // Get current cell id <curr_cell_id>, relative to <new_particle_coord_cell_id>
                    // Parallel check for intersect in <curr_cell_id>, passing <ordered_array>,
                    //    start index and end index for the <ordered_array> for current cell
                    
                    // Kernel function saves result in array of bools <intersects>
                    // Check the array <intersects>
                }

        bool not_intersected = true;
        if (not_intersected) {
            particle.x = x;
            particle.y = y;
            particle.z = z;
        }
    }
}

enum paramsMLen{
    TYPE_MLEN = 6, SN_MLEN = 5, NAME_MLEN = 4, ALT_LOC_IND_MLEN = 1, RES_NAME_MLEN = 3,
    CHAIN_IND_MLEN = 1, RES_SEQ_NUM_MLEN = 4, RES_INS_CODE_MLEN = 1,
    X_MLEN = 8, Y_MLEN = 8, Z_MLEN = 8, OCC_MLEN = 6, TEMP_FACTOR_MLEN = 6,
    SEG_ID_MLEN = 4, ELEM_SYMB_MLEN = 2, CHARGE_MLEN = 2
};

static std::string
format(double fp_num, unsigned nint, unsigned nfrac) {
    auto maxNum = std::pow(10, nint);
    if (fp_num >= maxNum)
        throw std::invalid_argument(std::string("Number is too big (max ")
                                    + std::to_string(maxNum) + std::string(")"));

    fp_num = std::ceil(fp_num * maxNum) / static_cast<double>(maxNum);

    std::stringstream fp_num_ss;
    fp_num_ss.precision(nfrac);
    fp_num_ss.setf(std::ios::fixed, std::ios::floatfield);
    fp_num_ss << fp_num;

    return fp_num_ss.str();
}

constexpr auto COORD_MINT = 4;
constexpr auto COORD_MFRAC = 3;

constexpr auto OCCTEMP_MINT = 3;
constexpr auto OCCTEMP_MFRAC = 2;

static std::string
fcoord (double coord) {
    return format(coord, COORD_MINT, COORD_MFRAC);
}

static std::string
focctemp (double occtemp) {
    return format(occtemp, OCCTEMP_MINT, OCCTEMP_MFRAC);
}

enum direction{left, right};

static std::string check_fill (std::string val, size_t len, direction align) {
    auto val_len = val.size();
    if (val_len == 0)
        for (auto i = len; i > 0; i--, val += " ");
    else if (val_len > len)
        throw std::invalid_argument("Invalid argument length (too long): expected " +
                                    std::to_string(len) + ", got " + std::to_string(val_len));
    else {
        std::string xfix;
        for (auto i = val.size(); i < len; i++, xfix += " ");
        val = (align == right) ? xfix + val : val + xfix;
    }
    return val;
}

static std::string
check_fill(std::string val, int len) {
    return check_fill(val, len, left);
}

static void
export_to_pdb ( std::string fn,             // output filename with extension
                std::string type,           // 1-6
                std::string sn,             // 7-11  right
                std::string name,           // 13-16
                std::string alt_loc_ind,    // 17
                std::string res_name,       // 18-20 right
                std::string chain_ind,      // 22
                std::string res_seq_num,    // 23-26 right
                std::string res_ins_code,   // 27
                std::string x,              // 31-38 right
                std::string y,              // 39-46 right
                std::string z,              // 47-54 right
                std::string occ,            // 55-60 right
                std::string temp_factor,    // 61-66 right
                std::string seg_id,         // 73-76
                std::string elem_symb,      // 77-78 right
                std::string charge          // 79-80
              ){

    type = check_fill(type, TYPE_MLEN);
    sn = check_fill(sn, SN_MLEN, right);
    name = check_fill(name, NAME_MLEN);
    alt_loc_ind = check_fill(alt_loc_ind, ALT_LOC_IND_MLEN);
    res_name = check_fill(res_name, RES_NAME_MLEN, right);
    chain_ind = check_fill(chain_ind, CHAIN_IND_MLEN);
    res_seq_num = check_fill(res_seq_num, RES_SEQ_NUM_MLEN, right);
    res_ins_code = check_fill(res_ins_code, RES_INS_CODE_MLEN);
    x = check_fill(x, X_MLEN, right);
    y = check_fill(y, Y_MLEN, right);
    z = check_fill(z, Z_MLEN, right);
    occ = check_fill(occ, OCC_MLEN, right);
    temp_factor = check_fill(temp_factor, TEMP_FACTOR_MLEN, right);
    seg_id = check_fill(seg_id, SEG_ID_MLEN);
    elem_symb = check_fill(elem_symb, ELEM_SYMB_MLEN, right);
    charge = check_fill(charge, CHARGE_MLEN);

    std::ofstream pdb_file(fn, std::ofstream::app);
    pdb_file << type << sn << " " << name << alt_loc_ind << res_name << " " << chain_ind
             << res_seq_num << res_ins_code << "   " << x << y << z << occ << temp_factor
             << "     " << elem_symb << charge << std::endl;
    pdb_file.close();
}

void Grid::export_to_pdb(std::string fn) {
    remove(fn.data());
    unsigned serial_num = 1;
    for (auto particle : particles) {

        std::string sn_str = std::to_string(serial_num);

        const std::string particle_type = "ATOM";
        const std::string atom_name = "C";
        const std::string sort_of_elem = std::to_string(1);
        const std::string temp_factor = focctemp(0);

        ::export_to_pdb(fn, particle_type, std::to_string(serial_num), atom_name, "", "", "", sort_of_elem, "",
                fcoord(particle.x), fcoord(particle.y), fcoord(particle.z),
                focctemp(particle.sigma), temp_factor, "", "", "");
        serial_num++;
    }
}
