// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <fstream>
#include <cmath>
#include "parse_config.cuh"
#include "time_measurement.cuh"
#include "grid.cuh"
#include "utils.cuh"


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Wrong arguments!" << std::endl;
        exit(1);
    }

    std::ifstream data(argv[1]);
    if (!data) {
        std::cout << "Error while opening the config file!" << std::endl;
        exit(3);
    }
    auto config = parse_conf(data);

    auto conf = Config::from_map(config);

    Grid grid(conf.Lx, conf.Ly, conf.Lz, D3<uint>{conf.N_cells}, conf.N);

    auto start_fill = get_current_time_fenced();
    grid.fill();
    auto finish_fill = get_current_time_fenced();
    grid.export_to_pdb("fill.pdb");

    auto start_move = get_current_time_fenced();
    grid.move(conf.dispmax);
    auto finish_move = get_current_time_fenced();
    grid.export_to_pdb("move.pdb");

    std::cout << "--------- Radial distibution function ---------" << std::endl;
    std::vector<double> rdf_vals;

    const double dr = 0.1;
    const double rmax = grid.get_Lx() / 2;

    auto rdf = compute_rdf(grid, dr, rmax);

    save_rdf_to_file(rdf, dr, rmax, "rdf.dat");

    //for (auto rdf_current : rdf) {
        //std::cout << rdf_current << std::endl;
    //}


    std::cout << "fill: " << to_us(finish_fill - start_fill) << " us\n" << "move: " << to_us(finish_move - start_move) << " us" << std::endl;

    return 0;
}

