// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <chrono>
#include <fstream>
#include <atomic>
#include "parse_config.h"
#include "time_measurement.h"
#include "grid.h"


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

    Grid grid(conf.Lx, conf.Ly, conf.Lz);

    auto start = get_current_time_fenced();
    grid.fill(conf.N);
    auto finish = get_current_time_fenced();


    std::cout << to_us(finish - start) << std::endl;

    return 0;
}

