// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <fstream>
#include <cmath>
#include "parse_config.cuh"
#include "time_measurement.cuh"
#include "grid.cuh"
#include "utils.cuh"


//__global__ void check_intersect_global(Particle *particle, Particle *particles) {

//}

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

    const double dr = 0.1;
    const double rmax = grid.L.x / 2;


    auto start_fill = get_current_time_fenced();
    grid.fill();
    auto finish_fill = get_current_time_fenced();
    grid.export_to_pdb("fill.pdb");

    auto rdf1 = compute_rdf(grid, dr, rmax);
    save_rdf_to_file(rdf1, dr, rmax, "rdf_fill.dat");


    auto start_move = get_current_time_fenced();
    grid.move(conf.dispmax);
    auto finish_move = get_current_time_fenced();
    grid.export_to_pdb("move.pdb");

    auto rdf2 = compute_rdf(grid, dr, rmax);
    save_rdf_to_file(rdf2, dr, rmax, "rdf_move.dat");


    //std::cout << "--------- Radial distibution function ---------" << std::endl;
    //for (auto rdf_current : rdf) {
        //std::cout << rdf_current << std::endl;
    //}


    std::cout << "fill: " << to_us(finish_fill - start_fill) << " us"
              << "\t=  "  << to_s(finish_fill - start_fill) << " s"
              << std::endl
              << "move: " << to_us(finish_move - start_move) << " us"
              << "\t=  "  << to_s(finish_move - start_move) << " s"
              << std::endl;

    return 0;
}

