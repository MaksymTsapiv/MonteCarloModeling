// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <random>
#include <vector>
#include <iostream>
#include "grid.h"
#include "particle.h"
#include "utils.h"


void Grid::common_initializer(int x, int y, int z){
    cells.reserve(dim_cells * dim_cells * dim_cells);

    for (int i = 0; i < dim_cells; i++) {
        for (int j = 0; j < dim_cells; j++) {
            for (int k = 0; k < dim_cells; k++) {
                cells.push_back(Cell(i, j, k));
            }
        }
    }

    Lx = x;
    Ly = y;
    Lz = z;
    compute_adj_cells();
}

Grid::Grid(double x, double y, double z) {
    common_initializer(x, y, z);
}

Grid::Grid(double x, double y, double z, int dim_cells_) {
    dim_cells = dim_cells_;
    common_initializer(x, y, z);
}


double random_double(double from, double to) {
    std::random_device rd;
    std::mt19937 rand_double(rd());

    std::uniform_real_distribution<> dist(from, to);
    return dist(rand_double);
}


int Grid::get_cell_id(int x, int y, int z) const {
    return x*dim_cells*dim_cells + y*dim_cells + z;
};

double Grid::get_Lx() const{
    return Lx;
}
double Grid::get_Ly() const{
    return Ly;
}
double Grid::get_Lz() const{
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


Particle Grid::get_particle(int id) const {
    for (auto particle : particles) {
        if (particle.get_id() == id) {
            return particle;
        }
    }
    return Particle();
}


double Grid::get_volume() const {
    return Lx * Ly * Lz;
}

size_t Grid::get_num_particles() const {
    return particles.size();
}

double Grid::get_density() const {
    return get_num_particles() / get_volume();
}

double Grid::distance(int id1, int id2) const {
    auto x_dist = std::min(fabs(get_particle(id1).get_x() - get_particle(id2).get_x()),
            Lx - fabs(get_particle(id1).get_x() - get_particle(id2).get_x()));

    auto y_dist = std::min(fabs(get_particle(id1).get_y() - get_particle(id2).get_y()),
            Ly - fabs(get_particle(id1).get_y() - get_particle(id2).get_y()));

    auto z_dist = std::min(fabs(get_particle(id1).get_z() - get_particle(id2).get_z()),
            Lz - fabs(get_particle(id1).get_z() - get_particle(id2).get_z()));

    return sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
}


std::vector<Particle> Grid::get_particles() const {
    return particles;
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
                if (distance(pid, particle.get_id()) <= sigma) {
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

void Grid::move(double dispmax) {
    bool not_intersected = true;
    size_t count = 0;
    double sigma = 1.0;

    for (size_t j = 0; j < particles.size(); j++) {
        double new_x = particles[j].get_x() + random_double(-1, 1);
        double new_y = particles[j].get_y() + random_double(-1, 1);
        double new_z = particles[j].get_z() + random_double(-1, 1);

        double vec_x = new_x - particles[j].get_x();
        double vec_y = new_y - particles[j].get_y();
        double vec_z = new_z - particles[j].get_z();

        double vec_length = sqrt(pow(vec_x, 2) + pow(vec_y, 2) + pow(vec_z, 2));

        vec_x = vec_x / vec_length;
        vec_y = vec_y / vec_length;
        vec_z = vec_z / vec_length;

        double x = particles[j].get_x() + vec_x * dispmax;
        double y = particles[j].get_y() + vec_y * dispmax;
        double z = particles[j].get_z() + vec_z * dispmax;

        if (x >= Lx) x -= Lx;
        if (y >= Ly) y -= Ly;
        if (z >= Lz) z -= Lz;

        if (x < 0) x += Lx;
        if (y < 0) y += Ly;
        if (z < 0) z += Lz;

        for (auto &cell : cells) {
            bool exit = false;
            for (auto pid : cell.get_particles()) {
                if (get_particle(pid).get_id() == particles[j].get_id())
                    continue;

                if (calc_dist(get_particle(pid), Particle(x, y, z, sigma)) <= sigma) {
                    not_intersected = false;
                    exit = true;
                    count++;
                    break;
                }
            }
            if (exit)
                break;
        }

        if (not_intersected) {
            particles[j].set_x(x);
            particles[j].set_y(y);
            particles[j].set_z(z);
        }
        not_intersected = true;
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

static std::string
check_fill (std::string val, size_t len, direction align) {
    auto val_len = val.size();
    if (val_len == 0)
        for (auto i = len; i > 0; i--, val += " ");
    else if (val_len > len)
        throw std::invalid_argument("Invalid argument length (too long): expected " +
                                    std::to_string(len) + ", got " + std::to_string(val_len));
    else {
        std::string xfix = "";
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
std::map<int, std::vector<int>> Grid::compute_adj_cells() {

    std::map<int, std::vector<int>> adj_cells;

    for (auto i = 0; i < dim_cells; i++)
    {
        int li = i == 0 ? dim_cells - 1 : i - 1;
        int ri = i == dim_cells - 1 ? 0 : i+1;
        for (auto j = 0; j < dim_cells; j++)
        {
            int lj = j == 0 ? dim_cells - 1 : j - 1;
            int rj = j == dim_cells - 1 ? 0 : j+1;
            for (auto k = 0; k < dim_cells; k++)
            {
                int lk = k == 0 ? dim_cells - 1 : k - 1;
                int rk = k == dim_cells - 1 ? 0 : k+1;

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
    //for (auto it = adj_cells.begin(); it != adj_cells.end(); ++it) {
        //std::cout << it->first << ": ";
        //for (auto it2 = it->second.begin(); it2 != it->second.end(); ++it2) {
            //std::cout << *it2 << " ";
        //}
        //std::cout << std::endl;
    //}

    return adj_cells;
}
