// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <filesystem>

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
    grid.setTemp(conf.temp);
    grid.print_grid_info();

    const double dr = 0.1;
    const double rmax = grid.L.x / 2;



    std::string dataDirname {"data"};
    std::string rdfDirPath = dataDirname + "/rdf";
    std::string cfDirPath = dataDirname + "/cf";
    std::string pdbDirPath = dataDirname + "/pdb";

    std::filesystem::remove_all(dataDirname);

    auto mkdirStatus = std::filesystem::create_directory(dataDirname);
    if (!mkdirStatus)
        throw std::runtime_error("Failed creating directory for data " + dataDirname);

    if (conf.rdf_step) {
        // TODO: Is / crossplatform?
        auto mkdirStatus = std::filesystem::create_directory(rdfDirPath);
        if (!mkdirStatus)
            throw std::runtime_error("Failed creating directory for rdf: " + rdfDirPath);
    }

    if (conf.export_cf_step) {
        auto mkdirStatus = std::filesystem::create_directory(cfDirPath);
        if (!mkdirStatus)
            throw std::runtime_error("Failed creating directory for cf: " + cfDirPath);
    }

    if (conf.export_pdb_step) {
        auto mkdirStatus = std::filesystem::create_directory(pdbDirPath);
        if (!mkdirStatus)
            throw std::runtime_error("Failed creating directory for pdb: " + pdbDirPath);
    }



    std::cout << "Initializing..." << std::endl;
    size_t fill_res = 0;
    auto start_init = get_current_time_fenced();
    if (conf.restore)
        grid.import_from_cf("init.cf");
    else {
        fill_res = grid.fill();
        if (conf.export_cf_step)
            grid.export_to_cf(cfDirPath + "/init.cf");
    }
    auto finish_init = get_current_time_fenced();

    std::cout << "   Done initializing\n";

    if (!conf.restore)
        std::cout << "Fill tries: " << fill_res;

    std::cout << ". Time: " << to_us(finish_init - start_init) << " us"
              << "  ~  "  << to_s(finish_init - start_init) << " s" << std::endl << std::endl;

    grid.system_energy();
    std::cout << "Initial energy = " << std::setprecision(8) << grid.get_energy() / conf.N << std::endl;



    grid.dfs_cluster(conf.connect_dist);
    grid.cluster_info(conf.connect_dist);


    std::vector<double> prev_rdf;
    if (conf.rdf_step) {
        std::cout << "Calculating and saving initial RDF..." << std::endl;
        prev_rdf = compute_rdf(grid, dr, rmax);
        save_rdf_to_file(prev_rdf, dr, rmax, "rdf_init.dat");
        std::cout << "  Done" << std::endl;
    }

    auto start_loop = get_current_time_fenced();
    for (auto i = 1; i <= conf.N_steps; ++i) {
        std::cout << std::endl << "Step " << i << std::endl;

        std::cout << "Moving... " << std::endl;
        auto n_moved = grid.move(conf.dispmax);
        std::cout << "  Done, moved " << n_moved << std::endl;

        if (conf.export_cf_step && i % conf.export_cf_step == 0) {
            std::cout << "Exporting to custom format... " << std::endl;
            grid.export_to_cf(cfDirPath + "/step_" + std::to_string(i) + ".cf");
            std::cout << "  Done" << std::endl;
        }
        if (conf.export_pdb_step && i % conf.export_pdb_step == 0) {
            std::cout << "Exporting to pdb... " << std::endl;
            grid.export_to_pdb(pdbDirPath + "/step_" + std::to_string(i) + ".pdb");
            std::cout << "  Done" << std::endl;
        }
        if (conf.rdf_step && i % conf.rdf_step == 0) {
            std::cout << "Calculating and saving RDF... " << std::endl;
            auto rdf = compute_rdf(grid, dr, rmax, prev_rdf);
            prev_rdf = rdf;
            save_rdf_to_file(rdf, dr, rmax, rdfDirPath + "/rdf_step_" + std::to_string(i) + ".dat");
            std::cout << "  Done" << std::endl;
        }
        if (conf.energy_step && i % conf.energy_step == 0) {
            std::cout << "Energy = " << std::setprecision(8) << grid.get_energy() / conf.N << std::endl;
        }
        grid.cluster_info(conf.connect_dist);
    }
    auto finish_loop = get_current_time_fenced();

    std::cout << "Clusters at the end:" << std::endl;
    grid.check_cluster();

    std::cout << "Exporting final system to custom format and pdb... " << std::endl;
    if (conf.export_pdb_step)
        grid.export_to_pdb(pdbDirPath + "/final.pdb");

    if (conf.export_cf_step)
        grid.export_to_cf(cfDirPath + "/final.cf");
    std::cout << "  Done" << std::endl;

    if (conf.rdf_step) {
        std::cout << "Calculating and saving final RDF..." << std::endl;
        auto final_rdf = compute_rdf(grid, dr, rmax, prev_rdf);
        save_rdf_to_file(final_rdf, dr, rmax, "rdf_final.dat");
        std::cout << "  Done" << std::endl;
    }

    std::cout << "Final energy = " << std::setprecision(8) << grid.get_energy() / conf.N << std::endl;


    std::cout << "Loop time: " << to_us(finish_loop - start_loop) << " us"
              << "\t=  "  << to_s(finish_loop - start_loop) << " s"
              << std::endl;

    return 0;
}

