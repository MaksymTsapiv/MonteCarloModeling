// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <fstream>
#include "parse_config.cuh"
#include "time_measurement.cuh"
#include "grid.cuh"


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

    Grid grid(conf.Lx, conf.Ly, conf.Lz, conf.N);

    auto start1 = get_current_time_fenced();
    grid.fill();
    grid.export_to_pdb("fill.pdb");

    //auto start2 = get_current_time_fenced();
    //double dispmax = 0.2;
    //grid.move(dispmax);
    //grid.export_to_pdb("move.pdb");


    //auto finish = get_current_time_fenced();

    //std::cout << "fill: " << to_us(start2 - start1)<< "\n" << "move: " << to_us(finish - start2) << std::endl;

    return 0;
}

