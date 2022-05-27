// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>

#include "grid.cuh"
#include "particle.cuh"
#include "time_measurement.cuh"

//std::random_device rd;
//std::mt19937 gen(rd());
std::mt19937 gen(1);

double random_double(double from, double to) {
    std::uniform_real_distribution<> dist(from, to);
    return dist(gen);
}

int random_int(int from, int to) {
    std::uniform_int_distribution<> dist(from, to);
    return dist(gen);
}

std::vector<Particle> Grid::get_particles() const {
    return particles;
}

double Grid::volume() const {
    return L.x * L.y * L.z;
}

size_t Grid::de_facto_n() const {
    return particles.size();
}

double Grid::density() const {
    return n / volume();
}

double Grid::packing_fraction() const {
    return (n*M_PI*pow(p_sigma, 3)) / (6.0*volume());
}

void Grid::print_grid_info() const {
    std::cout << "Simulation box size:\t\t" << L.x << " x " << L.y << " x " << L.z
        << " (volume = " << volume() << ")"<< std::endl;
    std::cout << "Num of cells per dimention:\t"
        << dim_cells.x << ", " << dim_cells.y << ", " << dim_cells.z << std::endl;
    std::cout << "Cell size:\t\t\t"
        << cell_size.x << " x " << cell_size.y << " x " << cell_size.z << std::endl;
    std::cout << "Packing fraction:\t\t" << packing_fraction() << std::endl;
    std::cout << "Density:\t\t\t" << density() << std::endl;
    std::cout << "Expected number of particles:\t" << n << std::endl;
    std::cout << "Particle's sigma (diameter):\t" << p_sigma << std::endl << std::endl;
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
D3<int> Grid::get_cell(D3<T> p) const {
    D3<double> new_p = normalize<double>(p.toD3double());

    int c_x = static_cast<int>(floor( (new_p.x / L.x) * dim_cells.x) );
    int c_y = static_cast<int>(floor( (new_p.y / L.y) * dim_cells.y) );
    int c_z = static_cast<int>(floor( (new_p.z / L.z) * dim_cells.z) );
    D3<int> cell{c_x, c_y, c_z};
    return cell;
}

template <typename T>
size_t Grid::cell_id(D3<T> p) const {
    return p.x + p.y*dim_cells.y + p.z*dim_cells.z*dim_cells.z;
}


__device__ double device_min(double a, double b) {
    return a < b ? a : b;
}

std::vector<size_t>
Grid::check_intersect_cpu(Particle particle) {
    std::vector<size_t> res;
    for (Particle p: particles) {
        auto xd = fabs(particle.x - p.x) < L.x - fabs(particle.x - p.x) ?
                        fabs(particle.x - p.x) : L.x - fabs(particle.x - p.x);

        auto yd = fabs(particle.y - p.y) < L.y - fabs(particle.y - p.y) ?
                        fabs(particle.y - p.y) : L.y - fabs(particle.y - p.y);

        auto zd = fabs(particle.z - p.z) < L.z - fabs(particle.z - p.z) ?
                        fabs(particle.z - p.z) : L.z - fabs(particle.z - p.z);

        double dist = hypot(hypot(xd, yd), zd);
        auto this_cell_id = cell_id(get_cell(p.get_coord()));
        if (dist < particle.sigma)
            res.push_back(p.id);
    }
    return res;
}

/*
 * Useful for debug purposes only, when check_intersect on CUDA is no working correctly
 */
std::vector<size_t>
Grid::check_intersect_cpu(Particle particle, uint req_cell_id) {
    std::vector<size_t> res;
    for (Particle p: particles) {
        auto xd = fabs(particle.x - p.x) < L.x - fabs(particle.x - p.x) ?
                        fabs(particle.x - p.x) : L.x - fabs(particle.x - p.x);

        auto yd = fabs(particle.y - p.y) < L.y - fabs(particle.y - p.y) ?
                        fabs(particle.y - p.y) : L.y - fabs(particle.y - p.y);

        auto zd = fabs(particle.z - p.z) < L.z - fabs(particle.z - p.z) ?
                        fabs(particle.z - p.z) : L.z - fabs(particle.z - p.z);

        double dist = hypot(hypot(xd, yd), zd);
        auto this_cell_id = cell_id(get_cell(p.get_coord()));
        if (dist < particle.sigma && this_cell_id == req_cell_id)
            res.push_back(p.id);
    }
    return res;
}

/*
 * Yet another oversload of check_intersect_cpu that accepts particle_id and ignores check for
 *  intersect with that particle. Useful in move() method
 */
std::vector<size_t>
Grid::check_intersect_cpu(Particle particle, uint req_cell_id, uint particle_id) {
    std::vector<size_t> res;
    for (Particle p: particles) {
        auto xd = fabs(particle.x - p.x) < L.x - fabs(particle.x - p.x) ?
                        fabs(particle.x - p.x) : L.x - fabs(particle.x - p.x);

        auto yd = fabs(particle.y - p.y) < L.y - fabs(particle.y - p.y) ?
                        fabs(particle.y - p.y) : L.y - fabs(particle.y - p.y);

        auto zd = fabs(particle.z - p.z) < L.z - fabs(particle.z - p.z) ?
                        fabs(particle.z - p.z) : L.z - fabs(particle.z - p.z);

        double dist = hypot(hypot(xd, yd), zd);
        auto this_cell_id = cell_id(get_cell(p.get_coord()));
        if (dist < particle.sigma && this_cell_id == req_cell_id && p.id != particle_id)
            res.push_back(p.id);
    }
    return res;
}

__global__ void update_kernel(uint *cellStartIdx, size_t cell_idx, size_t N) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId > N)
        return;
    cellStartIdx[cell_idx + threadId]++;
}

/* TODO: Rewrite with __shared__ uint* array and using parallel summing (reduce) algorithm.
 *  It should be faster like that then atomicAdd.
 */
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


__global__ void energy_single_kernel(double* energy, const Particle* particle,
                                     const Particle *particles, const uint *cellStartIdx, uint curr_cell_id,
                                     const D3<double> *L, uint curr_part_id, size_t partInCell, size_t arr_size) {

    extern __shared__ double part_energy[];

    const double sqe = -1.0;
    const double sqw = 0.2;
    const double inf = 0x7f800000;

    uint startIdx = cellStartIdx[curr_cell_id];
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;

    auto xd = device_min( fabs(particle->x - particles[startIdx+idx].x),
                          L->x - fabs(particle->x - particles[startIdx+idx].x) );

    auto yd = device_min( fabs(particle->y - particles[startIdx+idx].y),
                          L->y - fabs(particle->y - particles[startIdx+idx].y) );

    auto zd = device_min( fabs(particle->z - particles[startIdx+idx].z),
                          L->z - fabs(particle->z - particles[startIdx+idx].z) );

    auto dist = hypot(hypot(xd, yd), zd);

    if ((dist >= particle->sigma) && (dist < particle->sigma + sqw))
        part_energy[idx] = sqe;
    else if (dist < particle->sigma) {
        if (curr_part_id == particles[startIdx+idx].id)
            part_energy[idx] = 0.0;
        else {
            part_energy[idx] = inf;
//            printf("Error, intersected. %lu with %lu (cell %i) -- dist = %f\n",
//                   particle->id, particles[startIdx+idx].id, curr_cell_id, dist);
        }
    }
    else
        part_energy[idx] = 0;

    __syncthreads();

    if (idx+partInCell < arr_size)
        part_energy[idx+partInCell] = 0;

    for (auto i = arr_size/2; i > 0; i/=2) {
        if (idx < i)
            part_energy[idx] += part_energy[idx + i];
        __syncthreads();
    }

    if (idx == 0)
        *energy = part_energy[0];
}


void Grid::fill() {
    size_t count_tries = 0;
    size_t max_tries = 10000 * n;

    while ((particles.size() < n) && count_tries < max_tries) {

        double x = L.x * random_double(0, 1);
        double y = L.y * random_double(0, 1);
        double z = L.z * random_double(0, 1);

        Particle particle = Particle(x, y, z, p_sigma);

        Particle *cuda_particle;
        cudaMalloc(&cuda_particle, sizeof(Particle));
        cudaMemcpy(cuda_particle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);

        D3<double> p_point = particle.get_coord();
        D3<int> p_cell = get_cell(p_point);

        bool intersected = false;

        for (auto z_off = -1; z_off <= 1; ++z_off) {
            for (auto y_off = -1; y_off <= 1; ++y_off) {
                for (auto x_off = -1; x_off <= 1; ++x_off) {
                    cudaMemset(intersectsCuda, 0, sizeof(int));

                    // TODO: do it better, without double
                    D3<double> offset = {x_off*cell_size.x, y_off*cell_size.y, z_off*cell_size.z};
                    uint curr_cell_id = cell_id(get_cell(p_point + offset));
                    /************************************/

                    // number of particles in cell
                    size_t partInCell = partPerCell[curr_cell_id];

                    if (partInCell == 0)
                        continue;

                    const Particle *cuda_ordered_particles = particles_ordered.get_array();

                    size_t threadsPerBlock = std::min(partInCell, MAX_BLOCK_THREADS);
                    size_t numBlocks = (partInCell + threadsPerBlock - 1) / threadsPerBlock;
                    check_intersect<<<numBlocks, threadsPerBlock>>>(cuda_particle,
                                                cuda_ordered_particles, cellStartIdx,
                                                curr_cell_id, cudaL, intersectsCuda);

                    int *intersects = new int;
                    cudaMemcpy(intersects, intersectsCuda, sizeof(int),
                                                            cudaMemcpyDeviceToHost);

                    if (*intersects > 0) {
                        intersected = true;
                        delete intersects;
                        break;
                    }

                    delete intersects;
                }
                if (intersected) break;
            }
            if (intersected) break;
        }

        if (!intersected) {
            complex_insert(particle);
            if (particle.id % 1000 == 0)
                std::cout << "Inserting " << particle.id << "'s" << std::endl;
        }
        else // If a particle wasn't inserted, do not increment Particle's nextId counter
            Particle::nextId--;

        count_tries++;
        cudaFree(cuda_particle);
    }
    if (n != de_facto_n())
        throw std::runtime_error("Actual number of particles <de_facto_n()> in grid\
                is not equal to desired number of particles <n> after fill");

    std::cout << "Tries: " << count_tries << std::endl;

    std::cout << std::endl;
}

__global__ void backward_move_kernel(uint *cellStartIdx, size_t new_cell_id, size_t N) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= N)
        return;
    cellStartIdx[new_cell_id+1 + threadId]++;
}

__global__ void forward_move_kernel(uint *cellStartIdx, size_t init_cell_id, size_t N) {
    size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= N)
        return;
    cellStartIdx[init_cell_id+1 + threadId]--;
}

/*
 * Overload of check_intersect that accepts another argument <curr_part_id>,
 *  to ignore checking with particle with that id. This overload is used in move function
 */
__global__ void
check_intersect (
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

void Grid::move(double dispmax) {
    uint success = 0;

    for (size_t j = 0; j < n; j++) {
        auto &i = particles[random_int(0, n-1)];

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

        Particle particle = Particle(x, y, z, p_sigma);

        Particle *init_particle;
        cudaMalloc(&init_particle, sizeof(Particle));
        cudaMemcpy(init_particle, &i, sizeof(Particle), cudaMemcpyHostToDevice);

        Particle *cuda_particle;
        cudaMalloc(&cuda_particle, sizeof(Particle));
        cudaMemcpy(cuda_particle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);

        D3<double> p_point = particle.get_coord();
        D3<int> p_cell = get_cell(p_point);
        size_t new_p_cell_id = cell_id(p_cell);

        bool intersected = false;
        bool accept = false;

        double init_en_total = 0;
        double new_en_total = 0;
        for (auto z_off = -1; z_off <= 1; ++z_off) {
            for (auto y_off = -1; y_off <= 1; ++y_off) {
                for (auto x_off = -1; x_off <= 1; ++x_off) {
                    cudaMemset(intersectsCuda, 0, sizeof(int));

                    // TODO: do it better, without double
                    D3<double> offset = {x_off*cell_size.x, y_off*cell_size.y, z_off*cell_size.z};
                    uint curr_cell_id = cell_id(get_cell(p_point + offset));
                    /************************************/

                    // number of particles in cell
                    size_t partInCell = partPerCell[curr_cell_id];

                    if (partInCell == 0)
                        continue;

                    const Particle *cuda_ordered_particles = particles_ordered.get_array();


                    size_t arr_size = pow(2, ceil(log2(partInCell)));
                    energy_single_kernel<<<1, partInCell, arr_size*sizeof(double)>>>(energyCuda,
                                                                                     init_particle, cuda_ordered_particles, cellStartIdx, curr_cell_id,
                                                                                     cudaL, curr_part_id, partInCell, arr_size);

                    auto* init_en = new double;
                    cudaMemcpy(init_en, energyCuda, sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemset(energyCuda, 0, sizeof(double));

                    energy_single_kernel<<<1, partInCell, arr_size*sizeof(double)>>>(energyCuda,
                                                                                     cuda_particle, cuda_ordered_particles, cellStartIdx, curr_cell_id,
                                                                                     cudaL, curr_part_id, partInCell, arr_size);

                    init_en_total += *init_en;

                    auto* en = new double;
                    cudaMemcpy(en, energyCuda, sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemset(energyCuda, 0, sizeof(double));

                    if (*en > 0) {
                        intersected = true;
                        break;
                    }

                    new_en_total += *en;
                }
                if (intersected) break;
            }
            if (intersected) break;
        }

        auto delta_en = init_en_total - new_en_total;
        if (delta_en < 0) {
            accept = true;
        } else {
            if ((double) rand() / RAND_MAX < exp(-beta * delta_en))
                accept = true;
        }

        if (!intersected && accept) {
            i.x = particle.x;
            i.y = particle.y;
            i.z = particle.z;

            if (new_p_cell_id == init_p_cell_id)
                particles_ordered.update_particle(i.id, i);
            else {
                // Cell start index in ordered array for the current particle (which is inserted)
                uint *partCellStartIdx = new uint;
                cudaMemcpy(partCellStartIdx, &cellStartIdx[new_p_cell_id], sizeof(uint),
                           cudaMemcpyDeviceToHost);

                partPerCell[new_p_cell_id]++;
                partPerCell[init_p_cell_id]--;

                int remove_status = particles_ordered.remove_by_id(i.id);
                if (remove_status)
                    throw std::runtime_error("Error in remove");

                int insert_status = particles_ordered.insert(i, *partCellStartIdx);
                if (insert_status)
                    throw std::runtime_error("Error in insert");

                size_t cells_in_range = init_p_cell_id > new_p_cell_id ?
                            init_p_cell_id - new_p_cell_id : new_p_cell_id - init_p_cell_id;

                size_t threadsPerBlock = std::min(cells_in_range, MAX_BLOCK_THREADS);
                size_t numBlocks = (cells_in_range + threadsPerBlock - 1) / threadsPerBlock;

                if (init_p_cell_id > new_p_cell_id)
                    backward_move_kernel<<<numBlocks, threadsPerBlock>>>
                                (cellStartIdx, new_p_cell_id, cells_in_range);

                else if (init_p_cell_id < new_p_cell_id)
                    forward_move_kernel<<<numBlocks, threadsPerBlock>>>
                                (cellStartIdx, init_p_cell_id, cells_in_range);
            }
            success++;
        }

        cudaFree(cuda_particle);
    }
    std::cout << success << " moved" << std::endl;
}

void Grid::system_energy() {
    energy = 0;

    for (auto &particle: particles) {
        auto curr_part_id = particle.id;
        D3<double> p_point = particle.get_coord();

        Particle *cuda_particle;
        cudaMalloc(&cuda_particle, sizeof(Particle));
        cudaMemcpy(cuda_particle, &particle, sizeof(Particle), cudaMemcpyHostToDevice);

        const Particle *cuda_ordered_particles = particles_ordered.get_array();
        for (auto z_off = -1; z_off <= 1; ++z_off) {
            for (auto y_off = -1; y_off <= 1; ++y_off) {
                for (auto x_off = -1; x_off <= 1; ++x_off) {
                    D3<double> offset = {x_off*cell_size.x, y_off*cell_size.y, z_off*cell_size.z};
                    uint curr_cell_id = cell_id(get_cell(p_point + offset));

                    size_t partInCell = partPerCell[curr_cell_id];

                    if (partInCell == 0)
                        continue;
                    size_t arr_size = pow(2, ceil(log2(partInCell)));
                    energy_single_kernel<<<1, partInCell, arr_size*sizeof(double)>>>(energyCuda,
                            cuda_particle, cuda_ordered_particles, cellStartIdx, curr_cell_id,
                            cudaL, curr_part_id, partInCell, arr_size);

                    auto* en = new double;
                    cudaMemcpy(en, energyCuda, sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemset(energyCuda, 0, sizeof(double));

                    energy += *en;
                    delete en;
                }
            }
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

        ::export_to_pdb(fn, particle_type, std::to_string(serial_num), atom_name, "", "", "",
                sort_of_elem, "", fcoord(particle.x), fcoord(particle.y), fcoord(particle.z),
                focctemp(particle.sigma), temp_factor, "", "", "");
        serial_num++;
    }
}

void Grid::export_to_cf(const std::string& fn) {
    std::ofstream cf_file(fn);
    if (!cf_file)
        throw std::runtime_error("Error while opening file for export " + fn);

    for (auto p: particles) {
        char buff[256];
        sprintf(buff, "%15ld%15ld%4d%20.10lf%20.10lf%20.10lf\n", p.id, p.id, 1, p.x, p.y, p.z);
        std::string buff_str{buff};
        cf_file << buff_str;
    }
}

void Grid::import_from_cf(const std::string& fn) {
    // import from custom format file
    std::ifstream cf_file(fn);
    if (!cf_file)
        throw std::runtime_error("Error while opening file for import " + fn);

    std::string line;

    while (std::getline(cf_file, line)) {
        std::stringstream ss(line);

        // skip first 3 columns, because they are useless for particle constructor
        long trash;
        ss >> trash >> trash >> trash;

        double x, y, z;
        ss >> x >> y >> z;
        Particle p(x, y, z, p_sigma);
        complex_insert(p);
    }
    cf_file.close();

    if (particles.size() != n)
        throw std::invalid_argument("During import: too many particles in CF file.\
                Either grid is badly preconfigured or CF file is corrupted.");
}

void Grid::complex_insert(Particle p) {
    particles.push_back(p);
    auto p_cell_id = cell_id(get_cell(p.get_coord()));

    // Cell start index in ordered array for the current particle (which is inserted)
    uint *partCellStartIdx = new uint;
    cudaMemcpy(partCellStartIdx, &cellStartIdx[p_cell_id], sizeof(uint),
                                                cudaMemcpyDeviceToHost);

    particles_ordered.insert(p, *partCellStartIdx);
    partPerCell[p_cell_id]++;

    if (n_cells < p_cell_id + 1)
        throw std::runtime_error("Cell_idx > number of cells, which is impossible");

    size_t N = n_cells-p_cell_id-1;
    if (N > 0) {
        size_t threadsPerBlock = std::min(N, MAX_BLOCK_THREADS);
        size_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        update_kernel<<<numBlocks, threadsPerBlock>>>(cellStartIdx, p_cell_id+1, N);
    }

    free(partCellStartIdx);
}
