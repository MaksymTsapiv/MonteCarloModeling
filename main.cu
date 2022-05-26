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
    // TODO: Parse command line options
    if (argc != 2) {
        std::cout << "Wrong arguments!" << std::endl;
        exit(1);
    }

    std::ifstream data(argv[1]);
    if (!data)
        throw std::runtime_error("Error while opening the config file " + std::string(argv[1]));

    auto config = parse_conf(data);

    auto conf = Config::from_map(config);

    Grid grid(conf.Lx, conf.Ly, conf.Lz, D3<uint>{conf.N_cells}, conf.N);

    const double dr = 0.1;
    const double rmax = grid.L.x / 2;

    //grid.import_from_cf("fill.cf");
    //grid.move(conf.dispmax);
    //grid.export_to_pdb("move.pdb");
    //exit(1);

    auto start_fill = get_current_time_fenced();
    grid.fill();
    auto finish_fill = get_current_time_fenced();
    grid.export_to_pdb("fill.pdb");
    grid.export_to_cf("fill.cf");

    auto rdf1 = compute_rdf(grid, dr, rmax);
    save_rdf_to_file(rdf1, dr, rmax, "rdf_fill.dat");

    grid.system_energy();
    std::cout << "energy1 = " << grid.get_energy() << std::endl;


    auto start_move = get_current_time_fenced();
    grid.move(conf.dispmax);
    auto finish_move = get_current_time_fenced();
    grid.export_to_pdb("move.pdb");
    grid.export_to_cf("move.cf");

    auto rdf2 = compute_rdf(grid, dr, rmax);
    save_rdf_to_file(rdf2, dr, rmax, "rdf_move.dat");

    auto start_energy = get_current_time_fenced();
    grid.system_energy();
    auto finish_energy = get_current_time_fenced();

    std::cout << "energy2 = " << grid.get_energy() << std::endl;


    grid.system_energy();



    std::cout << "fill: " << to_us(finish_fill - start_fill) << " us"
              << "\t=  "  << to_s(finish_fill - start_fill) << " s"
              << std::endl;
    std::cout << "move: " << to_us(finish_move - start_move) << " us"
              << "\t=  "  << to_s(finish_move - start_move) << " s"
              << std::endl;
    std::cout << "energy: " << to_us(finish_energy - start_energy) << " us"
              << "\t=  "  << to_s(finish_energy - start_energy) << " s"
              << std::endl;

    return 0;
}

