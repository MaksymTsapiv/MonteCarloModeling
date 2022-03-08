// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <chrono>
#include <atomic>
#include "time_measurement.h"
#include "grid.h"
#include "particle.h"


int main(int argc, char* argv[]) {
    size_t n = std::stoi(argv[1]);
    double Lx = 10.0, Ly = 10.0, Lz = 10.0;
    Grid grid(Lx, Ly, Lz);

    auto start = get_current_time_fenced();
    grid.fill(n);
    auto finish = get_current_time_fenced();


    std::cout << to_us(finish - start) << std::endl;

    return 0;
}


