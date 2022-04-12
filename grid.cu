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
#include "cell.cuh"
#include "grid.cuh"
#include "particle.cuh"

__host__ __device__ double Grid::get_Lx() const{
    return Lx;
}
__host__ __device__ double Grid::get_Ly() const{
    return Ly;
}
__host__ __device__ double Grid::get_Lz() const{
    return Lz;
}

void Grid::set_Lx(double x) {
    Lx = x;
}
void Grid::set_Ly(double y) {
    Ly = y;
}
void Grid::set_Lz(double z) {
    Lz = z;
}

void Grid::common_initializer(double x, double y, double z){
    cells.reserve(dim_cells.x * dim_cells.y * dim_cells.z);

    for (int i = 0; i < dim_cells.x; i++) {
        for (int j = 0; j < dim_cells.y; j++) {
            for (int k = 0; k < dim_cells.z; k++) {
                cells.emplace_back(i, j, k);
            }
        }
    }

    Lx = x;
    Ly = y;
    Lz = z;
    adj_cells = compute_adj_cells();
}

double random_double(double from, double to) {
    std::random_device rd;
    std::mt19937 rand_double(rd());

    std::uniform_real_distribution<> dist(from, to);
    return dist(rand_double);
}


__host__ __device__ double calc_dist(Particle p1, Particle p2) {
    double x1 = p1.get_x();
    double x2 = p2.get_x();
    double y1 = p1.get_y();
    double y2 = p2.get_y();
    double z1 = p1.get_z();
    double z2 = p2.get_z();

    return hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
    // return sqrt(pow(sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2)), 2), pow((z1 -z2), 2));
}

size_t Grid::get_cell_id(double x, double y, double z) const {
    auto x_cell = static_cast<size_t>(floor(x / Lx * dim_cells.x));
    auto y_cell = static_cast<size_t>(floor(y / Ly * dim_cells.y));
    auto z_cell = static_cast<size_t>(floor(z / Lz * dim_cells.z));

    return x_cell + y_cell * dim_cells.y + z_cell * dim_cells.z * dim_cells.z;
};

__host__ __device__ Particle Grid::get_particle(size_t id) {
    for (auto particle : particles) {
        if (particle.get_id() == id) {
            return particle;
        }
    }
    return {};
}

void Grid::fill(size_t n) {
    bool flag = true;
    size_t count_tries = 0;
    size_t max_tries = 10000 + n;

    double sigma = 1.0;

    while ((particles.size() < n) && count_tries < max_tries) {

        double x = Lx * random_double(0, 1);
        double y = Ly * random_double(0, 1);
        double z = Lz * random_double(0, 1);

        Particle particle = Particle(x, y, z, sigma);

        for (auto &cell : cells) {
            for (auto pid : cell.get_particles()) {
                if (calc_dist(get_particle(pid), particle) <= sigma) {
                    flag = false;
                    break;
                }
            }
        }

        if (flag) {
            particles.push_back(particle);
            cells[get_cell_id(x, y, z)].add_particle(particle.get_id());
        }
        flag = true;

        count_tries++;
    }
}

// This is temporary function just to make it work. TODO: make up a better design
__device__ double calc_dist(double x1, double y1, double z1, double x2, double y2, double z2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));
}

__global__ void parent_kernel(Particle *p1, Particle *p, D3 grid_size, bool *intersects) {
    if (p1->get_id() == p[threadIdx.x].get_id()) {
        intersects[threadIdx.x] = false;
        return;
    }

    auto sigma = p1->get_sigma();

    auto x1 = p1->get_x();
    auto y1 = p1->get_y();
    auto z1 = p1->get_z();

    auto x2 = p[threadIdx.x].get_x();
    auto y2 = p[threadIdx.x].get_y();
    auto z2 = p[threadIdx.x].get_z();

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
        double new_x = particle.get_x() + random_double(-1, 1);
        double new_y = particle.get_y() + random_double(-1, 1);
        double new_z = particle.get_z() + random_double(-1, 1);

        double vec_x = new_x - particle.get_x();
        double vec_y = new_y - particle.get_y();
        double vec_z = new_z - particle.get_z();

        double vec_length = sqrt(pow(vec_x, 2) + pow(vec_y, 2) + pow(vec_z, 2));

        vec_x = vec_x / vec_length;
        vec_y = vec_y / vec_length;
        vec_z = vec_z / vec_length;

        double x = particle.get_x() + vec_x * dispmax;
        double y = particle.get_y() + vec_y * dispmax;
        double z = particle.get_z() + vec_z * dispmax;

        if (x >= Lx) x -= Lx;
        if (y >= Ly) y -= Ly;
        if (z >= Lz) z -= Lz;

        if (x < 0) x += Lx;
        if (y < 0) y += Ly;
        if (z < 0) z += Lz;

        std::vector<Particle> all_particles;
        for (auto cell_id : adj_cells[get_cell_id(particle.get_x(), particle.get_y(), particle.get_z())]) {
            const auto &cell = cells[cell_id];
            for (auto pid : cell.get_particles()) {
                if (get_particle(pid).get_id() == particle.get_id())
                    continue;

                all_particles.push_back(get_particle(pid));
            }
        }

        Particle particle_new = Particle(x, y, z, sigma);
        Particle *particle_cuda;
        cudaMalloc(&particle_cuda, sizeof(Particle));
        cudaMemcpy(particle_cuda, &particle_new, sizeof(Particle), cudaMemcpyHostToDevice);

        auto p_size = all_particles.size();

        Particle *all_particles_cuda;
        cudaMalloc(&all_particles_cuda, sizeof(Particle) * p_size);
        cudaMemcpy(all_particles_cuda, all_particles.data(), sizeof(Particle) * p_size, cudaMemcpyHostToDevice);

        bool *intersect_status;
        cudaMalloc(&intersect_status, sizeof(bool)*p_size);

        parent_kernel<<<1, p_size>>>(particle_cuda, all_particles_cuda, dim_cells, intersect_status);

        bool *intersect_status_host = (bool *) malloc(sizeof(bool) * p_size);
        cudaMemcpy(intersect_status_host, intersect_status, sizeof(bool)*p_size, cudaMemcpyDeviceToHost);

        bool not_intersected = true;
        for (size_t i = 0; i < p_size; i++) {
            if (intersect_status_host[i]) {
                not_intersected = false;
                break;
            }
        }

        if (not_intersected) {
            particle.set_x(x);
            particle.set_y(y);
            particle.set_z(z);
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
                fcoord(particle.get_x()), fcoord(particle.get_y()), fcoord(particle.get_z()),
                focctemp(particle.get_sigma()), temp_factor, "", "", "");
        serial_num++;
    }
}

/*
 * Find and return map where keys are cells and values are adjacent cells (excluding the key cell)
 */
std::map<size_t, std::vector<size_t>> Grid::compute_adj_cells() const {

    std::map<size_t, std::vector<size_t>> adj_cells;

    for (auto i = 0; i < dim_cells.x; i++)
    {
        int li = i == 0 ? dim_cells.x - 1 : i - 1;
        int ri = i == dim_cells.x - 1 ? 0 : i+1;
        for (auto j = 0; j < dim_cells.y; j++)
        {
            int lj = j == 0 ? dim_cells.y - 1 : j - 1;
            int rj = j == dim_cells.y - 1 ? 0 : j+1;
            for (auto k = 0; k < dim_cells.z; k++)
            {
                int lk = k == 0 ? dim_cells.z - 1 : k - 1;
                int rk = k == dim_cells.z - 1 ? 0 : k+1;

                adj_cells[get_cell_id(i, j, k)].push_back(get_cell_id(i, j, k));    // self

                adj_cells[get_cell_id(li, j, k)].push_back(get_cell_id(i, j, k));   // left on x axis
                adj_cells[get_cell_id(i, lj, k)].push_back(get_cell_id(i, j, k));   // left on y axis
                adj_cells[get_cell_id(i, j, lk)].push_back(get_cell_id(i, j, k));   // left on z axis

                adj_cells[get_cell_id(ri, j, k)].push_back(get_cell_id(i, j, k));   // right on x axis
                adj_cells[get_cell_id(i, rj, k)].push_back(get_cell_id(i, j, k));   // right on y axis
                adj_cells[get_cell_id(i, j, rk)].push_back(get_cell_id(i, j, k));   // right on z axis

                adj_cells[get_cell_id(li, lj, k)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(li, j, lk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(i, lj, lk)].push_back(get_cell_id(i, j, k));

                adj_cells[get_cell_id(li, rj, k)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(li, j, rk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(ri, lj, k)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(ri, j, rk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(i, rj, lk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(i, lj, rk)].push_back(get_cell_id(i, j, k));

                adj_cells[get_cell_id(ri, rj, k)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(ri, j, rk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(i, rj, rk)].push_back(get_cell_id(i, j, k));

                adj_cells[get_cell_id(li, lj, lk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(ri, rj, rk)].push_back(get_cell_id(i, j, k));

                adj_cells[get_cell_id(li, lj, rk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(li, rj, lk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(ri, lj, lk)].push_back(get_cell_id(i, j, k));

                adj_cells[get_cell_id(ri, rj, lk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(ri, lj, rk)].push_back(get_cell_id(i, j, k));
                adj_cells[get_cell_id(li, rj, rk)].push_back(get_cell_id(i, j, k));
            }
        }
    }

    // 100 * 100 * 100 * 27 * 4

    // print ajd_cells map
//    for (auto & adj_cell : adj_cells) {
//        std::cout << adj_cell.first << ": ";
//        for (int & it2 : adj_cell.second) {
//            std::cout << it2 << " ";
//        }
//        std::cout << std::endl;
//    }

    return adj_cells;
}
