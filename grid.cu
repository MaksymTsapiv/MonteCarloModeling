// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <random>
#include <cmath>
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>

#include "grid.cuh"
#include "particle.cuh"
#include "time_measurement.cuh"

__host__ __device__ double Grid::get_Lx() const{
    return L.x;
}
__host__ __device__ double Grid::get_Ly() const{
    return L.y;
}
__host__ __device__ double Grid::get_Lz() const{
    return L.z;
}

__host__ __device__ D3<double> Grid::get_L() const {
    return L;
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

std::vector<Particle> Grid::get_particles() const {
    return particles;
}

Particle Grid::get_particle(uint id) const {
    for (auto p : particles) {
        if (p.id == id) {
            return p;
        }
    }
    return {};
}

double Grid::volume() const {
    return L.x * L.y * L.z;
}

size_t Grid::n_particles() const {
    return particles.size();
}

double Grid::density() const {
    return n_particles() / volume();
}

double Grid::distance(int id1, int id2) const {
    auto x_dist = std::min(fabs(get_particle(id1).x - get_particle(id2).x),
            L.x - fabs(get_particle(id1).x - get_particle(id2).x));

    auto y_dist = std::min(fabs(get_particle(id1).y - get_particle(id2).y),
            L.y - fabs(get_particle(id1).y - get_particle(id2).y));

    auto z_dist = std::min(fabs(get_particle(id1).z - get_particle(id2).z),
            L.z - fabs(get_particle(id1).z - get_particle(id2).z));

    return sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
}

__device__ double device_min(double a, double b) {
    return a < b ? a : b;
}

__global__ void
check_intersect (
        const Particle *particle,
        const Particle *ordered_particles,
        const uint *cellStartIdx,
        uint curr_cell_id,
        const D3<double> *L,
        int *intersects) {

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
        atomicAdd(intersects, 1);
}

__global__ void update_kernel(uint *cellStartIdx, size_t cell_idx) {
    cellStartIdx[cell_idx+threadIdx.x]++;
}

void Grid::fill() {
    size_t count_tries = 0;
    size_t max_tries = 10000 * n;

    double sigma = 1.0;

    long long fors_time = 0;
    long long add_time = 0;

    while ((particles.size() < n) && count_tries < max_tries) {

        double x = L.x * random_double(0, 1);
        double y = L.y * random_double(0, 1);
        double z = L.z * random_double(0, 1);

        Particle particle = Particle(x, y, z, sigma);

        Particle *cuda_particle;
        cudaMalloc(&cuda_particle, sizeof(Particle));
        cudaMemcpy(cuda_particle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);

        D3<double> p_point = particle.get_coord();
        D3<int> p_cell = get_cell(p_point);

        bool intersected = false;

        auto for_start = get_current_time_fenced();
        for (auto z_off = -1; z_off <= 1; ++z_off) {
            for (auto y_off = -1; y_off <= 1; ++y_off) {
                for (auto x_off = -1; x_off <= 1; ++x_off) {
                    D3<int> offset = {x_off, y_off, z_off};
                    uint curr_cell_id = cell_id(p_cell + offset);

                    // number of particles in cell
                    size_t partInCell = partPerCell[curr_cell_id];

                    if (partInCell == 0)
                        continue;

                    const Particle *cuda_ordered_particles = particles_ordered.get_array();
                    // TODO: Variable block size
                    check_intersect<<<1, partInCell>>>( cuda_particle, cuda_ordered_particles,
                                                cellStartIdx, curr_cell_id, cudaL, intersectsCuda );

                    int *intersects = new int;
                    cudaMemcpy(intersects, intersectsCuda, sizeof(int),
                                                            cudaMemcpyDeviceToHost);

                    if (*intersects > 0)
                        intersected = true;

                    cudaMemset(intersectsCuda, 0, sizeof(int));

                    delete intersects;
                }
                if (intersected) break;
            }
            if (intersected) break;
        }
        auto for_end = get_current_time_fenced();
        fors_time += to_us(for_end - for_start);

        auto add_start = get_current_time_fenced();
        if (!intersected) {
            particles.push_back(particle);
            if (particles.size() % 1000 == 0) std::cout << "size = " << particles.size() << '\n';
            auto cell_idx = cell_id(p_cell);

            // Cell start index in ordered array for the current particle (which is inserted)
            uint *partCellStartIdx = new uint;
            cudaMemcpy(partCellStartIdx, &cellStartIdx[cell_idx], sizeof(uint),
                                                        cudaMemcpyDeviceToHost);

            auto add_start2 = get_current_time_fenced();

            particles_ordered.insert(particle, *partCellStartIdx);
            partPerCell[cell_idx]++;

            auto add_end2 = get_current_time_fenced();

            // TODO: Variable block size
            if (static_cast<int>(n_cells-cell_idx-1) > 0)
                update_kernel<<<1, n_cells-cell_idx-1>>>(cellStartIdx, cell_idx+1);
        }
        auto add_end = get_current_time_fenced();
        add_time += to_us(add_end - add_start);

        count_tries++;
        cudaFree(cuda_particle);
    }
    std::cout << "Tries: " << count_tries << std::endl;

    std::cout << "Fors time: " << fors_time << std::endl;
    std::cout << "Add time:  " << add_time << std::endl << std::endl;

    std::cout << std::endl;
}

__global__ void
check_intersect_move (
        const Particle *particle,
        const Particle *ordered_particles,
        const uint *cellStartIdx,
        uint curr_cell_id,
        const D3<double> *L,
        int *intersects,
        uint curr_part_id) {

    uint startIdx = cellStartIdx[curr_cell_id];
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (curr_part_id == ordered_particles[startIdx+idx].id)
        return;

    auto xd = device_min( fabs(particle->x - ordered_particles[startIdx+idx].x),
                        L->x - fabs(particle->x - ordered_particles[startIdx+idx].x) );

    auto yd = device_min( fabs(particle->y - ordered_particles[startIdx+idx].y),
                        L->y - fabs(particle->y - ordered_particles[startIdx+idx].y) );

    auto zd = device_min( fabs(particle->z - ordered_particles[startIdx+idx].z),
                        L->z - fabs(particle->z - ordered_particles[startIdx+idx].z) );

    auto dist = hypot(hypot(xd, yd), zd);
    if (dist < particle->sigma)
        atomicAdd(intersects, 1);
}

__global__ void backward_move_kernel(uint *cellStartIdx, size_t new_cell_id) {
    cellStartIdx[new_cell_id+1 + threadIdx.x]++;
}

__global__ void forward_move_kernel(uint *cellStartIdx, size_t init_cell_id) {
    cellStartIdx[init_cell_id+1 + threadIdx.x]--;
}

__global__ void print_particles_kernel(Particle *particles, size_t n) {
    for (size_t i = 0; i < n; i++)
        printf("%lu ", particles[i].id);
    printf("\n");
}

void Grid::move(double dispmax) {
    double sigma = 1.0;
    uint success = 0;

    for (auto & i : particles) {
        auto curr_part_id = i.id;

        D3<int> init_p_cell = get_cell(i.get_coord());
        size_t init_p_cell_id = cell_id(init_p_cell);

        double new_x = i.x + random_double(-1, 1);
        double new_y = i.y + random_double(-1, 1);
        double new_z = i.z + random_double(-1, 1);

        double vec_x = new_x - i.x;
        double vec_y = new_y - i.y;
        double vec_z = new_z - i.z;

        double vec_length = sqrt(pow(vec_x, 2) + pow(vec_y, 2) + pow(vec_z, 2));

        vec_x = vec_x / vec_length;
        vec_y = vec_y / vec_length;
        vec_z = vec_z / vec_length;

        double x = i.x + vec_x * dispmax;
        double y = i.y + vec_y * dispmax;
        double z = i.z + vec_z * dispmax;

        Particle particle = Particle(x, y, z, sigma);

        Particle *cuda_particle;
        cudaMalloc(&cuda_particle, sizeof(Particle));
        cudaMemcpy(cuda_particle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);

        D3<double> p_point = particle.get_coord();
        D3<int> p_cell = get_cell(p_point);
        size_t new_p_cell_id = cell_id(p_cell);

        bool intersected = false;

        for (auto z_off = -1; z_off <= 1; ++z_off) {
            for (auto y_off = -1; y_off <= 1; ++y_off) {
                for (auto x_off = -1; x_off <= 1; ++x_off) {
                    D3<int> offset = {x_off, y_off, z_off};
                    uint curr_cell_id = cell_id(p_cell + offset);

                    // number of particles in cell
                    size_t partInCell = partPerCell[curr_cell_id];

                    if (partInCell == 0)
                        continue;

                    const Particle *cuda_ordered_particles = particles_ordered.get_array();
                    // TODO: Variable block size
                    check_intersect_move<<<1, partInCell>>>(cuda_particle, cuda_ordered_particles,
                                                       cellStartIdx, curr_cell_id, cudaL,
						       intersectsCuda, curr_part_id);

                    int *intersects = new int;
                    cudaMemcpy(intersects, intersectsCuda, sizeof(int),
                               cudaMemcpyDeviceToHost);

                    if (*intersects > 0)
                        intersected = true;

                    cudaMemset(intersectsCuda, 0, sizeof(int));

                    delete intersects;
                }
                if (intersected) break;
            }
            if (intersected) break;
        }

        if (!intersected) {
            i.x = particle.x;
            i.y = particle.y;
            i.z = particle.z;
//            auto cell_idx = cell_id(p_cell);

            // Cell start index in ordered array for the current particle (which is inserted)
            uint *partCellStartIdx = new uint;
            cudaMemcpy(partCellStartIdx, &cellStartIdx[new_p_cell_id], sizeof(uint),
                       cudaMemcpyDeviceToHost);

            partPerCell[new_p_cell_id]++;
            partPerCell[init_p_cell_id]--;


            Particle curr_particle = i;
            int remove_status = particles_ordered.remove_by_id(i.id);
            if (remove_status)
                throw std::runtime_error("Error in remove");

            int insert_status = particles_ordered.insert(curr_particle, *partCellStartIdx);
            if (insert_status)
                throw std::runtime_error("Error in insert");

            uint cells_in_range = init_p_cell_id > new_p_cell_id ?
                        init_p_cell_id - new_p_cell_id : new_p_cell_id - init_p_cell_id;

            if (init_p_cell_id > new_p_cell_id) {
                // TODO: Variable block size
                backward_move_kernel<<<1, cells_in_range>>>(cellStartIdx, new_p_cell_id);
            }
            else if (init_p_cell_id < new_p_cell_id) {
                // TODO: Variable block size
                forward_move_kernel<<<1, cells_in_range>>>(cellStartIdx, init_p_cell_id);
            }
            success++;
        }

        cudaFree(cuda_particle);
    }
    std::cout << success << " moved" << std::endl;
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
    return check_fill(std::move(val), len, left);
}

static void
export_to_pdb ( const std::string& fn,             // output filename with extension
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

void Grid::export_to_pdb(const std::string& fn) {
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

/*
 * Expects that constructor has already been called, number of cells per dimention and grid size
 * are set
 */
void Grid::import_from_pdb(const std::string& fn) {
    std::ifstream pdb_file(fn);
    std::string line;
    while (std::getline(pdb_file, line)) {
        if (line.substr(0, 4) == "ATOM") {
            std::string x_str = line.substr(30, 8);
            std::string y_str = line.substr(38, 8);
            std::string z_str = line.substr(46, 8);
            std::string occ_str = line.substr(54, 6);

            double x = std::stod(x_str);
            double y = std::stod(y_str);
            double z = std::stod(z_str);
            double occ = std::stod(occ_str);

            particles.emplace_back(x, y, z, occ);
        }
    }
    pdb_file.close();

    if (particles.size() > n)
        throw std::invalid_argument("Too many particles in PDB file.\
                Either grid is badly preconfigured or PDB file is corrupted.");

    std::vector<Particle> sorted_particles;
    sorted_particles.reserve(particles.size());
    for (auto particle : particles)
        sorted_particles.push_back(particle);

    std::sort(sorted_particles.begin(), sorted_particles.end(), [](const Particle &a, const Particle &b) {
        return a.x < b.x || (a.x == b.x && a.y < b.y) || (a.x == b.x && a.y == b.y && a.z < b.z);
    });

    particles_ordered.set_data(sorted_particles.data(), sorted_particles.size());
}
