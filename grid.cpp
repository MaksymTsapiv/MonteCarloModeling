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
#include "grid.h"
#include "particle.h"


double random_double(double from, double to) {
    std::random_device rd;
    std::mt19937 rand_double(rd());

    std::uniform_real_distribution<> dist(from, to);
    return dist(rand_double);
}


double calc_dist(Particle p1, Particle p2) {
    double x1 = p1.get_x();
    double x2 = p2.get_x();
    double y1 = p1.get_y();
    double y2 = p2.get_y();
    double z1 = p1.get_z();
    double z2 = p2.get_z();

    return hypot(hypot(x1 - x2, y1 - y2), z1 - z2);
}

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

void Grid::fill(size_t n) {
    bool flag = true;
    size_t count_tries = 0;
    size_t max_tries = 10000 + n;

    double sigma = 1.0;

    double x = (Lx - sigma) * random_double(0, 1);
    double y = (Ly - sigma) * random_double(0, 1);
    double z = (Lz - sigma) * random_double(0, 1);

    Particle particle = Particle(x, y, z, sigma);
    particles.push_back(particle);

    while ((particles.size() != n) && count_tries != max_tries) {

        x = (Lx - sigma) * random_double(0, 1);
        y = (Ly - sigma) * random_double(0, 1);
        z = (Lz - sigma) * random_double(0, 1);

        particle = Particle(x, y, z, sigma);

        for (Particle p : particles) {
            if (calc_dist(p, particle) < sigma) {  //TODO: < or <= ???
                flag = false;
                break;
            }
        }

        if (flag) particles.push_back(particle);
        flag = true;

        count_tries++;
    }
}

void Grid::move() {
    bool flag = true;
    size_t count = 0;
    double dispmax = 0.2;   // TODO: unhardcode

    for (size_t j = 0; j < particles.size(); j++) {
        double x = particles[j].get_x() + random_double(0, dispmax);
        double y = particles[j].get_y() + random_double(0, dispmax);
        double z = particles[j].get_z() + random_double(0, dispmax);


        for (size_t i = 0; i < particles.size(); i++) {
            if (i == j)
                continue;
            if (calc_dist(Particle(x, y, z, dispmax), particles[i]) < dispmax) {
                flag = false;
                count++;
                break;
            }
        }
        if (flag) {
            particles[j].set_x(x);
            particles[j].set_y(y);
            particles[j].set_z(z);
        }
        flag = true;
    }
}

enum paramsMLen{
    TYPE_MLEN = 6, SN_MLEN = 5, NAME_MLEN = 4, ALT_LOC_IND_MLEN = 1, RES_NAME_MLEN = 3,
    CHAIN_IND_MLEN = 1, RES_SEQ_NUM_MLEN = 4, RES_INS_CODE_MLEN = 1,
    X_MLEN = 8, Y_MLEN = 8, Z_MLEN = 8, OCC_MLEN = 6, TEMP_FACTOR_MLEN = 6,
    SEG_ID_MLEN = 4, ELEM_SYMB_MLEN = 2, CHARGE_MLEN = 2
};

//typedef unsigned short param_len;

//constexpr param_len TYPE_MLEN;
//constexpr param_len SN_MLEN = 5;
//constexpr param_len NAME_MLEN = 4;
//constexpr param_len ALT_LOC_IND_MLEN = 1;
//constexpr param_len RES_NAME_MLEN = 3;
//constexpr param_len CHAIN_IND_MLEN = 1;
//constexpr param_len RES_SEQ_NUM_MLEN = 4;
//constexpr param_len RES_INS_CODE_MLEN = 1;
//constexpr param_len X_MLEN = 8;
//constexpr param_len Y_MLEN = 8;
//constexpr param_len Z_MLEN = 8;
//constexpr param_len OCC_MLEN = 6;
//constexpr param_len TEMP_FACTOR_MLEN = 6;
//constexpr param_len SEG_ID_MLEN = 4;
//constexpr param_len ELEM_SYMB_MLEN = 2;
//constexpr param_len CHARGE_MLEN = 2;

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

constexpr auto OCC_MINT = 3;
constexpr auto OCC_MFRAC = 2;

static std::string
fcoord (double coord) {
    return format(coord, COORD_MINT, COORD_MFRAC);
}

static std::string
focc (double occ) {
    return format(occ, OCC_MINT, OCC_MFRAC);
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
        ::export_to_pdb(fn, "ATOM", std::to_string(serial_num), "", "", "", "", "", "",
                fcoord(particle.get_x()), fcoord(particle.get_y()), fcoord(particle.get_z()),
                focc(particle.get_sigma()), "", "", "", "");
        serial_num++;
    }
}
