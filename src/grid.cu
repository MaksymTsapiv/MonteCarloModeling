// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>
#include <map>
#include <stack>
#include <algorithm>

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
    return (n*M_PI*pow(pSigma, 3)) / (6.0*volume());
}

void Grid::print_grid_info() const {
    std::cout << "Simulation box size:\t\t" << L.x << " x " << L.y << " x " << L.z
        << " (volume = " << volume() << ")"<< std::endl;
    std::cout << "Num of cells per dimention:\t"
        << dimCells.x << " x " << dimCells.y << " x " << dimCells.z << "  = "
        << n_cells << std::endl;
    std::cout << "Cell size:\t\t\t"
        << cellSize.x << " x " << cellSize.y << " x " << cellSize.z << std::endl;
    std::cout << "Average particles per cell:\t" << static_cast<double>(n)/n_cells << std::endl;
    std::cout << "Max particles per cell:\t\t" << maxPartPerCell << std::endl;
    std::cout << "Packing fraction:\t\t" << packing_fraction() << std::endl;
    std::cout << "Density:\t\t\t" << density() << std::endl;
    std::cout << "Temperature:\t\t\t" << temp << std::endl;
    std::cout << "Expected number of particles:\t" << n << std::endl;
    std::cout << "Particle's sigma (diameter):\t" << pSigma << std::endl << std::endl;
}

template <typename T>
D3<T> Grid::normalize(const D3<T> p) const {
    D3<T> new_p = p;

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
D3<int> Grid::cell(D3<T> p) const {
    D3<double> new_p = normalize<double>(p.toD3double());

    int c_x = static_cast<int>(floor( (new_p.x / L.x) * dimCells.x) );
    int c_y = static_cast<int>(floor( (new_p.y / L.y) * dimCells.y) );
    int c_z = static_cast<int>(floor( (new_p.z / L.z) * dimCells.z) );
    D3<int> cell{c_x, c_y, c_z};
    return cell;
}

template <typename T>
size_t Grid::cell_id(D3<T> p) const {
    return p.x + p.y*dimCells.y + p.z*dimCells.z*dimCells.z;
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
        auto this_cell_id = cell_id(cell(p.get_coord()));
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
        auto this_cell_id = cell_id(cell(p.get_coord()));
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
        auto this_cell_id = cell_id(cell(p.get_coord()));
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


__global__ void energy_all_cell_kernel(double* energy, Particle particle, const uint *partPerCell,
                                       const Particle *particles, const uint *cellStartIdx,
                                       const D3<double> *L,
                                       const AdjCells *adjCells, uint currPartCell)
{
    extern __shared__ double part_energy[];

    const double sqe = -1.0;
    const double sqw = 0.2;
    const double inf = 0x7f800000;

    uint blockI = blockIdx.x;
    if (blockI >= 27)
        return;

    uint currCellId = adjCells[currPartCell].ac[blockI];
    uint startIdx = cellStartIdx[currCellId];

    if (threadIdx.x < partPerCell[currCellId]) {
        auto xd = device_min( fabs(particle.x - particles[startIdx+threadIdx.x].x),
                              L->x - fabs(particle.x - particles[startIdx+threadIdx.x].x) );

        auto yd = device_min( fabs(particle.y - particles[startIdx+threadIdx.x].y),
                              L->y - fabs(particle.y - particles[startIdx+threadIdx.x].y) );

        auto zd = device_min( fabs(particle.z - particles[startIdx+threadIdx.x].z),
                              L->z - fabs(particle.z - particles[startIdx+threadIdx.x].z) );

        auto dist = hypot(hypot(xd, yd), zd);

        if ((dist >= particle.sigma) && (dist < particle.sigma + sqw))
            part_energy[threadIdx.x] = sqe;
        else if (dist < particle.sigma) {
            if (particle.id == particles[startIdx+threadIdx.x].id)
                part_energy[threadIdx.x] = 0.0;
            else {
                part_energy[threadIdx.x] = inf;
                printf("Error, intersected. %lu with %lu (cell %i) -- dist = %f\n",
                       particle.id, particles[startIdx+threadIdx.x].id, currCellId, dist);
            }
        }
        else
            part_energy[threadIdx.x] = 0;
    }
    else
        part_energy[threadIdx.x] = 0;

    __syncthreads();

    for (auto i = blockDim.x/2; i > 0; i/=2) {
        if (threadIdx.x < i)
            part_energy[threadIdx.x] += part_energy[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        energy[blockI] = part_energy[0];

}




size_t Grid::fill() {
    size_t count_tries = 0;
    size_t max_tries = 10000 * n;

    while ((particles.size() < n) && count_tries < max_tries) {

        double x = L.x * random_double(0, 1);
        double y = L.y * random_double(0, 1);
        double z = L.z * random_double(0, 1);

        Particle particle = Particle(x, y, z, pSigma);

        auto pCellId = cell_id(cell(particle.get_coord()));

        bool intersected = false;
        double energy_loc = 0.0;

        energy_all_cell_kernel<<<nAdjCells, maxPartPerCell2pow, maxPartPerCell2pow*sizeof(double)>>>
                        (energiesCuda, particle, partPerCellCuda, orderedParticlesCuda.get_array(),
                         cellStartIdxCuda, cudaL, cnCuda, pCellId);

        auto *energies = new double[nAdjCells];
        cudaMemcpy(energies, energiesCuda, sizeof(double) * nAdjCells, cudaMemcpyDeviceToHost);

        for (uint i = 0; i < nAdjCells; i++)
            energy_loc += energies[i];

        if (energy_loc > 0)
            intersected = true;

        if (!intersected) {
            complex_insert(particle);
            if (particle.id % 1000 == 0)
                std::cout << "Inserting " << particle.id << "'s" << std::endl;
        }
        else // If a particle wasn't inserted, do not increment Particle's nextId counter
            Particle::nextId--;

        count_tries++;
    }
    if (n != de_facto_n())
        throw std::runtime_error("Actual number of particles <de_facto_n()> in grid\
                is not equal to desired number of particles <n> after fill");

    cudaMemcpy(partPerCellCuda, partPerCell, sizeof(uint)*n_cells, cudaMemcpyHostToDevice);

    return count_tries;
}



void Grid::dfs_cluster(double connectDist) {
    std::vector<int> in_cluster(static_cast<int>(n), 0);
    std::stack<size_t> pidStack;

    uint* cellStartIdx = new uint[n_cells];
    Particle* orderedParticles = new Particle[n];

    cudaMemcpy(cellStartIdx, cellStartIdxCuda, sizeof(uint) * n_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(orderedParticles, orderedParticlesCuda.get_array(),
                sizeof(Particle) * n, cudaMemcpyDeviceToHost);


    for (auto i = 0; i < n; ++i) {
        if (in_cluster[i] > 0)
            continue;
        const auto particle = particles[i];

        pidStack.push(particle.id);
        in_cluster[particle.id]++;

        while (!pidStack.empty()) {
            const auto part = particles[pidStack.top()];
            const auto parentClusterId = part.clusterId;
            pidStack.pop();

            D3<double> p_point = part.get_coord();

            for (auto z_off = -1; z_off <= 1; ++z_off) {
                for (auto y_off = -1; y_off <= 1; ++y_off) {
                    for (auto x_off = -1; x_off <= 1; ++x_off) {
                        cudaMemset(intersectsCuda, 0, sizeof(int));
                        D3<double> offset = {x_off*cellSize.x, y_off*cellSize.y, z_off*cellSize.z};
                        size_t curr_cell_id = cell_id(cell(p_point + offset));

                        size_t partInCell = partPerCell[curr_cell_id];

                        if (partInCell == 0)
                            continue;

                        uint nextCellStartIdx;
                        if (curr_cell_id+1 == n_cells)
                            nextCellStartIdx = n_cells;
                        else
                            nextCellStartIdx = cellStartIdx[curr_cell_id+1];

                        for (int j = cellStartIdx[curr_cell_id]; j < nextCellStartIdx; ++j) {
                            auto &currPart = orderedParticles[j];
                            if (currPart.id == part.id)
                                continue;

                            auto xd = std::min( fabs(part.x - currPart.x),
                                    L.x - fabs(part.x - currPart.x) );

                            auto yd = std::min( fabs(part.y - currPart.y),
                                    L.y - fabs(part.y - currPart.y) );

                            auto zd = std::min( fabs(part.z - currPart.z),
                                    L.z - fabs(part.z - currPart.z) );

                            auto dist = hypot(hypot(xd, yd), zd);

                            if (dist <= connectDist && !in_cluster[currPart.id]) {
                                pidStack.push(currPart.id);
                                in_cluster[currPart.id]++;
                                orderedParticles[j].clusterId = parentClusterId;
                                particles[j].clusterId = parentClusterId;
                            }
                        }
                    }
                }
            }
        }
    }

    orderedParticlesCuda.set_data(orderedParticles, n);
}


void Grid::check_cluster() {
    std::map<size_t, size_t> clusters;
    for (auto & particle : particles)
        clusters[particle.clusterId]++;

    for (auto & cluster : clusters)
        std::cout << cluster.first << " : " << cluster.second << '\n';
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



/*
 * Update cluster id of particles1 & particles2. It is supposed to be called on all particles
 *  in the Grid. It will check if particle is in the cluster that should be changed and
 *  updates such particles's cluster id.
 */
__global__ void update_parts_cluster_kernel(Particle *particles1, Particle *particles2,
                const size_t* clusterIds, size_t nUniqueClusters, size_t newClusterId, size_t N)
{
    uint threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= N)
        return;

    bool returnFlag = false;
    // Check if current particle's clusterId is in clusterIds array, so that it should be changed
    for (size_t i = 0; i < nUniqueClusters; i++) {
        if (clusterIds[i] == particles1[threadId].clusterId) {
            particles1[threadId].clusterId = newClusterId;
            if (returnFlag)
                return;
            returnFlag = true;
        }
        if (clusterIds[i] == particles2[threadId].clusterId) {
            particles2[threadId].clusterId = newClusterId;
            if (returnFlag)
                return;
            returnFlag = true;
        }
    }
}



__global__ void energy_and_cluster_kernel(double* energy, Particle particle, const uint *partPerCell,
                                       const Particle *particles, const uint *cellStartIdx,
                                       const D3<double> *L, const AdjCells *adjCells,
                                       uint currPartCell, int *clusters)
{
    extern __shared__ double part_energy[];

    const double sqe = -1.0;
    const double sqw = 0.2;
    const double inf = 0x7f800000;

    uint blockI = blockIdx.x;
    if (blockI >= 27)
        return;

    uint currCellId = adjCells[currPartCell].ac[blockI];
    uint startIdx = cellStartIdx[currCellId];

    clusters[blockDim.x * blockI + threadIdx.x] = -1;

    if (threadIdx.x < partPerCell[currCellId]) {
        auto xd = device_min( fabs(particle.x - particles[startIdx+threadIdx.x].x),
                              L->x - fabs(particle.x - particles[startIdx+threadIdx.x].x) );

        auto yd = device_min( fabs(particle.y - particles[startIdx+threadIdx.x].y),
                              L->y - fabs(particle.y - particles[startIdx+threadIdx.x].y) );

        auto zd = device_min( fabs(particle.z - particles[startIdx+threadIdx.x].z),
                              L->z - fabs(particle.z - particles[startIdx+threadIdx.x].z) );

        auto dist = hypot(hypot(xd, yd), zd);

        if ((dist >= particle.sigma) && (dist < particle.sigma + sqw)) {
            part_energy[threadIdx.x] = sqe;
            clusters[blockDim.x * blockI + threadIdx.x] = particles[startIdx+threadIdx.x].clusterId;
            // printf("   clusters[%i] = %lu\n", blockDim.x * blockI + threadIdx.x, particles[startIdx+threadIdx.x].clusterId);
        }
        else if (dist < particle.sigma) {
            if (particle.id == particles[startIdx+threadIdx.x].id)
                part_energy[threadIdx.x] = 0.0;
            else {
                part_energy[threadIdx.x] = inf;
            }
        }
        else
            part_energy[threadIdx.x] = 0;
    }
    else
        part_energy[threadIdx.x] = 0;

    __syncthreads();

    for (auto i = blockDim.x/2; i > 0; i/=2) {
        if (threadIdx.x < i)
            part_energy[threadIdx.x] += part_energy[threadIdx.x + i];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        energy[blockI] = part_energy[0];

}

__global__ void print_some_parts(const Particle *particles, const uint *cellStartIdx) {
    // for (int i = 0; i < 64; i++){
    //     printf("Cell %i : %u\n", i, cellStartIdx[i]);
    // }
    // printf("\n\n");
    for (int i = 0; i < 100; i++){
        auto currPart = particles[cellStartIdx[0]+i];
        printf("%i) %lu: (%f, %f, %f)\n", i, currPart.id, currPart.x, currPart.y, currPart.z);
    }
    // for (int i = 0; i < 4; i++){
    //     auto currPart = particles[cellStartIdxCuda[11]+i];
    //     printf("%lu: (%f, %f, %f)\n", currPart.id, currPart.x, currPart.y, currPart.z);
    // }
}

size_t Grid::move(double dispmax, int step) {
    uint success = 0;

    /* Contains ids of clusters of particles that interact with the given particle.
     *  It should be int because if i-th particle doesn't interact with current particle
     *  clustersIdCuda[i] = -1
     */
    int *clustersIdCuda;
    auto clustersArrSizeBytes = nAdjCells * maxPartPerCell2pow * sizeof(int);
    cudaMalloc(&clustersIdCuda, clustersArrSizeBytes);

    // Capacity of shared memory. It will be used in configurations upon kernel function call
    auto sharedMemSizeBytes = maxPartPerCell2pow * sizeof(double);

    for (size_t _ = 0; _ < n; _++) {
        std::cout << "Cycle " << _ << std::endl;
        auto &currPart = particles[random_int(0, n-1)];

        D3<int> init_p_cell = cell(currPart.get_coord());
        size_t initPCellId = cell_id(init_p_cell);

        double new_x = currPart.x + random_double(-1, 1);
        double new_y = currPart.y + random_double(-1, 1);
        double new_z = currPart.z + random_double(-1, 1);

        double vec_x = new_x - currPart.x;
        double vec_y = new_y - currPart.y;
        double vec_z = new_z - currPart.z;

        double vec_length = sqrt(pow(vec_x, 2) + pow(vec_y, 2) + pow(vec_z, 2));

        vec_x = vec_x / vec_length;
        vec_y = vec_y / vec_length;
        vec_z = vec_z / vec_length;

        double x = currPart.x + vec_x * dispmax;
        double y = currPart.y + vec_y * dispmax;
        double z = currPart.z + vec_z * dispmax;

        Particle particle = Particle(x, y, z, pSigma);
        Particle::nextId--;     // Reset Particle::nextId, because <particle> is temporary particle
        particle.id = currPart.id;

        D3<double> p_point = particle.get_coord();
        size_t newPCellId = cell_id(cell(p_point));

        bool intersected = false;
        bool accept = false;

        if (_ == 78)
            auto jii = 0;

        energy_all_cell_kernel<<<nAdjCells, maxPartPerCell2pow, maxPartPerCell2pow*sizeof(double)>>>
                        (energiesCuda, currPart, partPerCellCuda, orderedParticlesCuda.get_array(),
                         cellStartIdxCuda, cudaL, cnCuda, initPCellId);

        auto *preEnergies = new double[nAdjCells];
        cudaMemcpy(preEnergies, energiesCuda, sizeof(double) * nAdjCells, cudaMemcpyDeviceToHost);

        double preEnergy = 0.0;
        for (uint k = 0; k < nAdjCells; k++)
            preEnergy += preEnergies[k];

        delete[] preEnergies;

        energy_and_cluster_kernel<<<nAdjCells, maxPartPerCell2pow, sharedMemSizeBytes>>>
                        (energiesCuda, particle, partPerCellCuda, orderedParticlesCuda.get_array(),
                         cellStartIdxCuda, cudaL, cnCuda, newPCellId, clustersIdCuda);

        auto *postEnergies = new double[nAdjCells];
        cudaMemcpy(postEnergies, energiesCuda, sizeof(double) * nAdjCells, cudaMemcpyDeviceToHost);

        double postEnergy = 0.0;
        for (uint k = 0; k < nAdjCells; k++)
            postEnergy += postEnergies[k];

        delete[] postEnergies;

        if (postEnergy > 0)
            intersected = true;

        auto delta_en = postEnergy - preEnergy;

        if (delta_en > 0) {
            if ((double) rand() / RAND_MAX < exp(-beta * delta_en))
                accept = true;
        } else {
            accept = true;
        }

        if (!intersected && accept) {
            energy += delta_en;
            currPart.x = particle.x;
            currPart.y = particle.y;
            currPart.z = particle.z;

            int *clusters = new int[nAdjCells * maxPartPerCell2pow];
            cudaMemcpy(clusters, clustersIdCuda, clustersArrSizeBytes, cudaMemcpyDeviceToHost);

            std::vector<size_t> uniqueClusters;

            for (int j = 0; j < nAdjCells * maxPartPerCell2pow; j++) {
                if (clusters[j] != -1)
                    if (std::find(uniqueClusters.begin(), uniqueClusters.end(), clusters[j])
                                                                        == uniqueClusters.end())
                        uniqueClusters.push_back(clusters[j]);
            }

            if (uniqueClusters.size() > 1) {
                size_t minClusterId = n;
                for (const auto &clusterId : uniqueClusters)
                    if (clusterId < minClusterId)
                        minClusterId = clusterId;

                // Remove minClusterId from uniqueClusters so that uniqueClusters
                //  contains only clusters that should be changed
                uniqueClusters.erase(std::find(uniqueClusters.begin(), uniqueClusters.end(),
                                                                                minClusterId));

                size_t threadsPerBlock = std::min(n, MAX_BLOCK_THREADS);
                size_t numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

                // uniqueClusters size in bytes
                size_t ucSizeBytes = sizeof(size_t) * uniqueClusters.size();
                size_t *uniqueClustersCuda;
                cudaMalloc(&uniqueClustersCuda, ucSizeBytes);
                cudaMemcpy(uniqueClustersCuda, uniqueClusters.data(), ucSizeBytes,
                                                            cudaMemcpyHostToDevice);

                /* Copying from host to device & then vice versa to update particles' cluster
                 *   in both host vector Grid::particles and device array orderedParticlesCuda
                 *   is a very bad idea, it should be super slow.
                 * TODO: Redesign it.   HOW?
                 */
                Particle *particlesCuda;
                cudaMalloc(&particlesCuda, n * sizeof(Particle));
                cudaMemcpy(particlesCuda, particles.data(), n * sizeof(Particle),
                                                            cudaMemcpyHostToDevice);

                update_parts_cluster_kernel<<<numBlocks, threadsPerBlock>>>
                            (orderedParticlesCuda.get_mutable_array(), particlesCuda, uniqueClustersCuda,
                             uniqueClusters.size(), minClusterId, n);

                Particle *particlesTmp = new Particle[n];
                cudaMemcpy(particlesTmp, particlesCuda, n * sizeof(Particle), cudaMemcpyDeviceToHost);
                particles = std::vector<Particle>{particlesTmp, particlesTmp + n};
            }

            if (newPCellId == initPCellId) {
                auto updateStatus = orderedParticlesCuda.update_particle(currPart.id, currPart);
                if (updateStatus)
                    throw std::runtime_error("Error in update_particle");
            }

            else {
                // Cell start index in ordered array for the current particle (which is inserted)
                uint *partCellStartIdx = new uint;
                cudaMemcpy(partCellStartIdx, &cellStartIdxCuda[newPCellId], sizeof(uint),
                           cudaMemcpyDeviceToHost);

                partPerCell[newPCellId]++;
                partPerCell[initPCellId]--;

                cudaMemcpy(&partPerCellCuda[newPCellId], &partPerCell[newPCellId],
                                                    sizeof(uint), cudaMemcpyHostToDevice);
                cudaMemcpy(&partPerCellCuda[initPCellId], &partPerCell[initPCellId],
                                                    sizeof(uint), cudaMemcpyHostToDevice);

                auto remove_status = orderedParticlesCuda.remove_by_id(currPart.id);
                if (remove_status)
                    throw std::runtime_error("Error in remove");

                auto insert_status = orderedParticlesCuda.insert(particle, *partCellStartIdx);
                if (insert_status)
                    throw std::runtime_error("Error in insert");

                size_t cells_in_range = initPCellId > newPCellId ?
                            initPCellId - newPCellId : newPCellId - initPCellId;

                size_t threadsPerBlock = std::min(cells_in_range, MAX_BLOCK_THREADS);
                size_t numBlocks = (cells_in_range + threadsPerBlock - 1) / threadsPerBlock;

                if (initPCellId > newPCellId)
                    backward_move_kernel<<<numBlocks, threadsPerBlock>>>
                                (cellStartIdxCuda, newPCellId, cells_in_range);

                else if (initPCellId < newPCellId)
                    forward_move_kernel<<<numBlocks, threadsPerBlock>>>
                                (cellStartIdxCuda, initPCellId, cells_in_range);

                delete partCellStartIdx;
            }
            success++;
            std::cout << "Accepting" << std::endl;
        }
        else
            std::cout << "Rejecting" << std::endl;
        std::cout << std::endl;
    }
    return success;
}

void Grid::system_energy() {
    energy = 0;

    for (auto &particle: particles) {
        D3<double> p_point = particle.get_coord();
        auto pCellId = cell_id(cell(p_point));

        energy_all_cell_kernel<<<nAdjCells, maxPartPerCell2pow, maxPartPerCell2pow*sizeof(double)>>>
                        (energiesCuda, particle, partPerCellCuda, orderedParticlesCuda.get_array(),
                         cellStartIdxCuda, cudaL, cnCuda, pCellId);

        auto *energies = new double[nAdjCells];
        cudaMemcpy(energies, energiesCuda, sizeof(double) * nAdjCells, cudaMemcpyDeviceToHost);

        for (uint i = 0; i < nAdjCells; i++)
            energy += energies[i];

        delete[] energies;
    }
    energy /= 2.0;
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
    // Workaround
    if (stoi(sn) >= 100000)
        sn = "99999";

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
        Particle p(x, y, z, pSigma);
        complex_insert(p);
    }
    cf_file.close();

    if (particles.size() != n)
        throw std::invalid_argument("During import: too many particles in CF file.\
                Either grid is badly preconfigured or CF file is corrupted.");
}

void Grid::complex_insert(Particle p) {
    particles.push_back(p);
    auto p_cell_id = cell_id(cell(p.get_coord()));

    // Cell start index in ordered array for the current particle (which is inserted)
    uint *partCellStartIdx = new uint;
    cudaMemcpy(partCellStartIdx, &cellStartIdxCuda[p_cell_id], sizeof(uint),
                                                cudaMemcpyDeviceToHost);

    orderedParticlesCuda.insert(p, *partCellStartIdx);
    partPerCell[p_cell_id]++;
    cudaMemcpy(&partPerCellCuda[p_cell_id], &partPerCell[p_cell_id], sizeof(uint),
                                                            cudaMemcpyHostToDevice);

    if (n_cells < p_cell_id + 1)
        throw std::runtime_error("Cell_idx > number of cells, which is impossible");

    size_t N = n_cells-p_cell_id-1;
    if (N > 0) {
        size_t threadsPerBlock = std::min(N, MAX_BLOCK_THREADS);
        size_t numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
        update_kernel<<<numBlocks, threadsPerBlock>>>(cellStartIdxCuda, p_cell_id+1, N);
    }

    delete partCellStartIdx;
}




void Grid::compute_adj_cells() {
    for (auto ix = 0; ix < dimCells.x; ix++) {
        for (auto iy = 0; iy < dimCells.y; iy++) {
            for (auto iz = 0; iz < dimCells.z; iz++) {
                auto ikx = ix;
                auto iky = iy;
                auto ikz = iz;
                auto parr_cell_id = cell_id(D3<int>(ikx, iky, ikz));
                auto k = 0;
                for (auto jx = ix-1; jx <= ix+1; jx++) {
                    for (auto jy = iy-1; jy <= iy+1; jy++) {
                        for (auto jz = iz-1; jz <= iz+1; jz++) {
                            // this cell coordinates
                            auto tc_x = jx;
                            auto tc_y = jy;
                            auto tc_z = jz;
                            if (tc_x < 0) tc_x += dimCells.x; if (tc_x > dimCells.x-1) tc_x -= dimCells.x;
                            if (tc_y < 0) tc_y += dimCells.y; if (tc_y > dimCells.y-1) tc_y -= dimCells.y;
                            if (tc_z < 0) tc_z += dimCells.z; if (tc_z > dimCells.z-1) tc_z -= dimCells.z;
                            
                            uint curr_cell_id = cell_id(D3<double>(tc_x, tc_y, tc_z));
                            cn[parr_cell_id].ac[k] = curr_cell_id;
                            k++;
                        }
                    }
                }
            }
        }
    }
}
