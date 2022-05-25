#ifndef PARSE_CONFIG_FILE_H
#define PARSE_CONFIG_FILE_H

#include <map>
#include <string>

std::map<std::string, double> parse_conf(std::ifstream &file);

class Config {
public:
    double Lx, Ly, Lz, dispmax;
    size_t N;
    uint N_cells;

    static Config from_map(std::map<std::string, double> &config);
};

#endif //PARSE_CONFIG_FILE_H
