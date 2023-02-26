#include <iostream>
#include <fstream>
#include <ostream>
#include <map>
#include <string>
#include <algorithm>
#include "parse_config.cuh"


std::pair<std::pair<std::vector<int>, Eigen::Matrix<double, Eigen::Dynamic, 3>>,
        std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>>>
        parse_patches(std::ifstream &file) {

    int patches_num;
    std::vector<int> types;
    std::vector<double> coors;
    std::pair<std::vector<int>, Eigen::Matrix<double, 4, 3>> patches;
    std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>> interaction;
    double temp1;
    int temp2;

    file >> patches_num;
    for (auto i = 0; i < patches_num; ++i) {
        for (auto j = 0; j < 3; ++j) {
            file >> temp1;
            coors.push_back(temp1);
        }
        file >> temp2;
        types.push_back(temp2);
    }
    Eigen::Matrix<double, 4, 3> mat {{coors[0], coors[1], coors[2]},
                                     {coors[3], coors[4], coors[5]},
                                     {coors[6], coors[7], coors[8]},
                                     {coors[9], coors[10], coors[11]}};

    patches = {types, mat};

    types.clear();
    coors.clear();
    for (auto k = 0; k < 3; ++k) {
        for (auto l = 0; l < 2; ++l) {
            file >> temp2;
            types.push_back(temp2);
        }
        for (auto l = 0; l < 2; ++l) {
            file >> temp1;
            coors.push_back(temp1);
        }
    }

    for (auto i = 0; i < 6; i+=2) {
        interaction.push_back({{types[i], types[i+1]}, {coors[i],coors[i+1]}});
    }


    return {patches, interaction};
}


std::map<std::string, double> parse_conf(std::ifstream &file) {
    std::string line{};
    std::string word{};
    std::map<std::string, double> conf;

    while (getline(file, line)) {
        if (line.empty()) continue;
         line.erase(remove_if(line.begin(), line.end(), isspace), line.end());
        auto temp = line.substr(0, line.find('#'));
        auto tmp = temp.find('=');
        conf[temp.substr(0, tmp)] = std::stod(temp.substr(tmp).erase(0, 1));
    }

    return conf;
}

Config Config::from_map(std::map<std::string, double> &config) {
    Config conf{};
    conf.Lx = config["Lx"];
    conf.Ly = config["Ly"];
    conf.Lz = config["Lz"];
    conf.N_steps = static_cast<size_t>(config["N_steps"]);
    conf.dispmax = config["dispmax"];
    conf.connect_dist = config["connect_dist"];
    conf.N_cells = static_cast<uint>(config["N_cells"]);
    conf.N = static_cast<size_t>(config["N"]);
    conf.restore = static_cast<size_t>(config["restore"]);

    conf.rdf_step = static_cast<size_t>(config["rdf_step"]);
    conf.export_pdb_step = static_cast<size_t>(config["export_pdb_step"]);
    conf.export_cf_step = static_cast<size_t>(config["export_cf_step"]);
    conf.energy_step = static_cast<size_t>(config["energy_step"]);
    conf.cluster_step = static_cast<size_t>(config["cluster_step"]);

    conf.temp = static_cast<size_t>(config["temp"]);
    return conf;
}
