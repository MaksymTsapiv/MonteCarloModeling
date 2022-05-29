#include <fstream>
#include <ostream>
#include <map>
#include <string>
#include <algorithm>
#include "parse_config.cuh"


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
    conf.N_cells = static_cast<uint>(config["N_cells"]);
    conf.N = static_cast<size_t>(config["N"]);
    conf.restore = static_cast<size_t>(config["restore"]);

    conf.rdf_step = static_cast<size_t>(config["rdf_step"]);
    conf.export_pdb_step = static_cast<size_t>(config["export_pdb_step"]);
    conf.export_cf_step = static_cast<size_t>(config["export_cf_step"]);
    conf.energy_step = static_cast<size_t>(config["energy_step"]);
    return conf;
}
