#ifndef PARSE_CONFIG_FILE_H
#define PARSE_CONFIG_FILE_H

#include <map>
#include <string>
#include <Eigen/Dense>

std::pair<std::pair<std::vector<int>, Eigen::Matrix<double, 4, 3>>, std::vector<std::pair<std::pair<int, int>, std::pair<double, double>>>> parse_patches(std::ifstream &file);

std::map<std::string, double> parse_conf(std::ifstream &file);

class Config {
public:
    double Lx{}, Ly{}, Lz{}, dispmax{}, connect_dist{};
    size_t N{}, N_steps{};
    uint N_cells{};

    bool restore = false;

    // 0 means never
    int rdf_step = 0;
    int export_pdb_step = 0;
    int export_cf_step = 0;
    int energy_step = 0;
    int cluster_step = 0;

    double temp = 1.0;

    static Config from_map(std::map<std::string, double> &config);
};

#endif //PARSE_CONFIG_FILE_H
