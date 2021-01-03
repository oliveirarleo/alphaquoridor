#ifndef QUORIDOR_MAPINFO_H
#define QUORIDOR_MAPINFO_H


#include <vector>


struct QuoridorMapInfo {
    int map_width;
    int map_height;
    std::vector<std::vector<int>> vwalls;
    std::vector<std::vector<int>> hwalls;
};


#endif