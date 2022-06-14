// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <fstream>
#include "parse_config.h"
#include "time_measurement.h"
#include "grid.h"
#include "utils.h"


int main(int argc, char* argv[]) {
    // TODO: set seed

    double dispmax = 0.2;
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

    Grid grid(conf.Lx, conf.Ly, conf.Lz, conf.N_cells);

    auto fill_start = get_current_time_fenced();
    grid.fill(conf.N);
    auto fill_end = get_current_time_fenced();
    std::cout << "start clusterization\n";
    grid.assign_cluster();
    std::cout << "check clusterization\n";
    grid.check_clusterization();

//    grid.export_to_pdb("fill.pdb");
    //std::cout << rdf(3.0, 1.9, grid) << std::endl;

//    auto move_start = get_current_time_fenced();
//    grid.move(dispmax);
//    auto move_end = get_current_time_fenced();

    //grid.export_to_pdb("move.pdb");
    //std::cout << rdf(3.0, 1.9, grid) << std::endl;

    std::cout << "Fill time: " << to_us(fill_end - fill_start) << " us\n";
//    << "Move time: " << to_us(move_end - move_start) << " us" << std::endl;

    return 0;
}

