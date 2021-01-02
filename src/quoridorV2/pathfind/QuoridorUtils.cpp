#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "QuoridorMapSearchNode.h"
#include "QuoridorMapInfo.h"

void printBoard(const std::vector<std::vector<int>> &board) {
    for (const auto &line : board) {
        for (const auto &e : line) {
            std::cout << e;
        }
        std::cout << '\n';
    }
    std::cout << '\n';

}

//// If is blocked North of player
//inline bool hasHWN(const std::vector<std::vector<int>> &hwalls, int x, int y, int board_size) {
//    return (y >= board_size) || (y<0) || (x < board_size && hwalls[x][y] == 1) || (x > 0 && hwalls[x-1][y] == 1);
//}
//
//// If is blocked East of player
//inline bool hasVWE(const std::vector<std::vector<int>> &vwalls, int x, int y, int board_size) {
//    return (x >= board_size) || (x<0) || (y < board_size && vwalls[x][y] == 1) || (y > 0 && vwalls[x][y-1] == 1);
//}

void setPawnActions(int player_x, int player_y, int opponent_x, int opponent_y,
                    const std::vector<std::vector<int>> &vwalls, const std::vector<std::vector<int>> &hwalls,
                    std::vector<int> &actions) {
    const int N = 0;
    const int S = 1;
    const int E = 2;
    const int W = 3;
    const int JN = 4;
    const int JS = 5;
    const int JE = 6;
    const int JW = 7;
    const int NE = 8;
    const int SW = 9;
    const int NW = 10;
    const int SE = 11;

    int board_size = (int) vwalls.size();
    //    NORTH
    // If nothing blocks north
    if (!hasHWN(hwalls, player_x, player_y, board_size)) {
        // If no player on north
        if ((player_x != opponent_x) || (player_y + 1 != opponent_y)) {
            actions[N] = 1;
        } else {
            // If nothing blocking jump north
            if (!hasHWN(hwalls, player_x, player_y + 1, board_size)) {
                actions[JN] = 1;
            } else {
                // If nothing blocking north east
                if (!hasVWE(vwalls, player_x, player_y + 1, board_size)) {
                    actions[NE] = 1;
                }
                // If nothing blocking north west
                if (!hasVWE(vwalls, player_x - 1, player_y + 1, board_size)) {
                    actions[NW] = 1;
                }
            }

        }
    }

    //    SOUTH
    // If nothing blocks south
    if (!hasHWN(hwalls, player_x, player_y - 1, board_size)) {
        // If no player on south
        if ((player_x != opponent_x) || (player_y - 1 != opponent_y)) {
            actions[S] = 1;
        } else {
            // If nothing blocking jump south
            if (!hasHWN(hwalls, player_x, player_y - 2, board_size)) {
                actions[JS] = 1;
            } else {
                // If nothing blocking south east
                if (!hasVWE(vwalls, player_x, player_y - 1, board_size)) {
                    actions[SE] = 1;
                }
                // If nothing blocking south west
                if (!hasVWE(vwalls, player_x - 1, player_y - 1, board_size)) {
                    actions[SW] = 1;
                }
            }

        }
    }

    //    EAST
    // If nothing blocks east
    if (!hasVWE(vwalls, player_x, player_y, board_size)) {
        // If no player on east
        if ((player_x + 1 != opponent_x) || (player_y != opponent_y)) {
            actions[E] = 1;
        } else {
            // If nothing blocking jump east
            if (!hasVWE(vwalls, player_x + 1, player_y, board_size)) {
                actions[JE] = 1;
            } else {
                // If nothing blocking north east
                if (!hasHWN(hwalls, player_x + 1, player_y, board_size)) {
                    actions[NE] = 1;
                }
                // If nothing blocking south east
                if (!hasHWN(hwalls, player_x + 1, player_y - 1, board_size)) {
                    actions[SE] = 1;
                }
            }

        }
    }

    //    WEST
    // If nothing blocks west
    if (!hasVWE(vwalls, player_x - 1, player_y, board_size)) {
        // If no player on west
        if ((player_x - 1 != opponent_x) || (player_y != opponent_y)) {
            actions[W] = 1;
        } else {
            // If nothing blocking jump west
            if (!hasVWE(vwalls, player_x - 2, player_y, board_size)) {
                actions[JW] = 1;
            } else {
                // If nothing blocking north west
                if (!hasHWN(hwalls, player_x - 1, player_y, board_size)) {
                    actions[NW] = 1;
                }
                // If nothing blocking south west
                if (!hasHWN(hwalls, player_x - 1, player_y - 1, board_size)) {
                    actions[SW] = 1;
                }
            }

        }
    }
}

std::vector<int> getPawnActions(int player_x, int player_y, int opponent_x, int opponent_y,
                                const std::vector<std::vector<int>> &vwalls,
                                const std::vector<std::vector<int>> &hwalls) {
    std::vector<int> actions(12, 0);
    setPawnActions(player_x, player_y, opponent_x, opponent_y, vwalls, hwalls, actions);
    return actions;
}

inline bool pathExists(
        int start_x,
        int start_y,
        int end_x,
        int end_y,
        const std::vector<std::vector<int>> &vwalls,
        const std::vector<std::vector<int>> &hwalls) {

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost
    // of travel across the terrain. Zero means the least possible difficulty
    // in travelling (think ice rink if you can skate) whilst 5 represents the
    // most difficult. 9 indicates that we cannot pass.

    // Create an instance of the search class...

    struct QuoridorMapInfo Map;
    Map.vwalls = vwalls;
    Map.hwalls = hwalls;
    Map.map_width = vwalls[0].size();
    Map.map_height = vwalls[0].size();

    AStarSearch<QuoridorMapSearchNode> astarsearch;

    // QuoridorMapSearchNode nodeStart;
    QuoridorMapSearchNode nodeStart = QuoridorMapSearchNode(start_x, start_y, Map);
    QuoridorMapSearchNode nodeEnd(end_x, end_y, Map);

    // Set Start and goal states
    astarsearch.SetStartAndGoalStates(nodeStart, nodeEnd);

    unsigned int SearchState;
    do {
        SearchState = astarsearch.SearchStep();
    } while (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_SEARCHING);


    if (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_SUCCEEDED) {
        astarsearch.FreeSolutionNodes();
        astarsearch.EnsureMemoryFreed();
        return true;

    }
    astarsearch.EnsureMemoryFreed();
    return false;
}


inline std::tuple<std::vector<int>, int> findPath(
        std::vector<int> &start,
        std::vector<int> &end,
        const std::vector<std::vector<int>> &vwalls,
        const std::vector<std::vector<int>> &hwalls) {

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost
    // of travel across the terrain. Zero means the least possible difficulty
    // in travelling (think ice rink if you can skate) whilst 5 represents the
    // most difficult. 9 indicates that we cannot pass.

    // Create an instance of the search class...

    struct QuoridorMapInfo Map;
    Map.vwalls = vwalls;
    Map.hwalls = hwalls;
    Map.map_width = vwalls[0].size();
    Map.map_height = vwalls[0].size();

    AStarSearch<QuoridorMapSearchNode> astarsearch;

    unsigned int SearchCount = 0;
    const unsigned int NumSearches = 1;

    // full path
    std::vector<int> path_full;
    // a short path only contains path corners
    std::vector<int> path_short;
    // how many steps used
    int steps = 0;

    while (SearchCount < NumSearches) {
        // QuoridorMapSearchNode nodeStart;
        QuoridorMapSearchNode nodeStart = QuoridorMapSearchNode(start[0], start[1], Map);
        QuoridorMapSearchNode nodeEnd(end[0], end[1], Map);

        // Set Start and goal states
        astarsearch.SetStartAndGoalStates(nodeStart, nodeEnd);

        unsigned int SearchState;
        unsigned int SearchSteps = 0;

        do {
            SearchState = astarsearch.SearchStep();
            SearchSteps++;
        } while (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_SEARCHING);

        if (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_SUCCEEDED) {
            // std::cout << "Search found goal state\n";
            QuoridorMapSearchNode *node = astarsearch.GetSolutionStart();
            steps = 0;

            // node->PrintNodeInfo();
            path_full.push_back(node->x);
            path_full.push_back(node->y);
            path_short.push_back(node->x);
            path_short.push_back(node->y);

            while (true) {
                node = astarsearch.GetSolutionNext();

                if (!node) {
                    break;
                }

                // node->PrintNodeInfo();
                path_full.push_back(node->x);
                path_full.push_back(node->y);

                steps++;

                /*
                Let's say there are 3 steps, x0, x1, x2. To verify whether x1 is a corner for the path.
                If the coordinates of x0 and x1 at least have 1 component same, and the coordinates of
                x0 and x2 don't have any components same, then x1 is a corner.
                If the path only contains 3 steps or less, path_full = path_short.
                */

                if (((path_full[2 * steps - 4] == path_full[2 * steps - 2]) ||
                     (path_full[2 * steps - 3] == path_full[2 * steps - 1])) &&
                    ((path_full[2 * steps - 4] != node->x) && (path_full[2 * steps - 3] != node->y)) && (steps > 2)) {
                    path_short.push_back(path_full[2 * steps - 2]);
                    path_short.push_back(path_full[2 * steps - 1]);
                }

            }

            // This works for both steps>2 and steps <=2
            path_short.push_back(path_full[path_full.size() - 2]);
            path_short.push_back(path_full[path_full.size() - 1]);

            // Once you're done with the solution you can free the nodes up
            astarsearch.FreeSolutionNodes();

        } else if (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_FAILED) {
            std::cout << "Search terminated. Did not find goal state\n";
        }

        SearchCount++;

        astarsearch.EnsureMemoryFreed();

    }

    return {path_full, steps};
}


bool canPlaceWall(int px, int py, int pgx, int pgy,
                  int ox, int oy, int ogx, int ogy,
                  int wx, int wy, bool is_vertical,
                  std::vector<std::vector<int>> &vwalls,
                  std::vector<std::vector<int>> &hwalls) {
    if (is_vertical)
        vwalls[wx][wy] = 1;
    else
        hwalls[wx][wy] = 1;

    bool res = pathExists(px, py, pgx, pgy, vwalls, hwalls) && pathExists(ox, oy, ogx, ogy, vwalls, hwalls);

    if (is_vertical)
        vwalls[wx][wy] = 0;
    else
        hwalls[wx][wy] = 0;

    return res;
}

void setWallActions(int px, int py, int pgx, int pgy,
                    int ox, int oy, int ogx, int ogy,
                    std::vector<std::vector<int>> &vwalls,
                    std::vector<std::vector<int>> &hwalls,
                    std::vector<std::vector<int>> &vwall_actions,
                    std::vector<std::vector<int>> &hwall_actions) {
    int board_size = (int) vwalls.size();
    for (int i = 0; i < board_size; ++i) {
        for (int j = 0; j < board_size; ++j) {
            // Vwalls
            if ((vwalls[i][j] == 0) &&
                (j + 1 >= board_size || vwalls[i][j + 1] == 0) &&
                (j - 1 < 0 || vwalls[i][j - 1] == 0) &&
                (hwalls[i][j] == 0)) {

                int connections = 0;
                if ((j + 1 >= board_size) ||
                    (hwalls[i][j + 1] == 1) ||
                    (i - 1 >= 0 && hwalls[i - 1][j + 1] == 1) ||
                    (i + 1 < board_size && hwalls[i + 1][j + 1] == 1) ||
                    (j + 2 < board_size && vwalls[i][j + 2] == 1))
                    connections += 1;
                if ((j - 1 < 0) ||
                    (hwalls[i][j - 1] == 1) ||
                    (i - 1 >= 0 && hwalls[i - 1][j - 1] == 1) ||
                    (i + 1 < board_size && hwalls[i + 1][j - 1] == 1) ||
                    (j - 2 >= 0 && vwalls[i][j - 2] == 1))
                    connections += 1;
                if ((i - 1 >= 0 && hwalls[i - 1][j] == 1) ||
                    (i + 1 < board_size && hwalls[i + 1][j] == 1))
                    connections += 1;

                if (connections < 2 ||
                    canPlaceWall(px, py, pgx, pgy, ox, oy, ogx, ogy, i, j, true, vwalls, hwalls)) {
                    vwall_actions[i][j] = 1;
                }
            }

            // Hwalls
            if ((hwalls[i][j] == 0) &&
                (i + 1 >= board_size || hwalls[i + 1][j] == 0) &&
                (i - 1 < 0 || hwalls[i - 1][j] == 0) &&
                (vwalls[i][j] == 0)) {

                int connections = 0;
                if ((i + 1 >= board_size) ||
                    (vwalls[i + 1][j] == 1) ||
                    (j - 1 >= 0 && vwalls[i + 1][j - 1] == 1) ||
                    (j + 1 < board_size && vwalls[i + 1][j + 1] == 1) ||
                    (i + 2 < board_size && hwalls[i + 2][j] == 1))
                    connections += 1;

                if ((i - 1 < 0) ||
                    (vwalls[i - 1][j] == 1) ||
                    (j - 1 >= 0 && vwalls[i - 1][j - 1] == 1) ||
                    (j + 1 < board_size && vwalls[i - 1][j + 1] == 1) ||
                    (i - 2 >= 0 && hwalls[i - 2][j] == 1))
                    connections += 1;

                if ((j - 1 >= 0 && vwalls[i][j - 1] == 1) ||
                    (j + 1 < board_size && vwalls[i][j + 1] == 1))
                    connections += 1;

                if (connections < 2 ||
                    canPlaceWall(px, py, pgx, pgy, ox, oy, ogx, ogy, i, j, false, vwalls, hwalls)) {
                    hwall_actions[i][j] = 1;
                }
            }
        }
    }
}

void updateWallActions_(int px, int py, int pgx, int pgy,
                        int ox, int oy, int ogx, int ogy,
                        std::vector<std::vector<int>> &vwalls,
                        std::vector<std::vector<int>> &hwalls,
                        const std::vector<std::vector<int>> &old_legal_vwall_actions,
                        const std::vector<std::vector<int>> &old_legal_hwall_actions,
                        std::vector<std::vector<int>> &vwall_actions,
                        std::vector<std::vector<int>> &hwall_actions) {
    int board_size = (int) vwalls.size();
    for (int i = 0; i < board_size; ++i) {
        for (int j = 0; j < board_size; ++j) {
            // Vwalls
            if ((vwalls[i][j] == 0 && old_legal_vwall_actions[i][j] == 1) &&
                (j + 1 >= board_size || vwalls[i][j + 1] == 0) &&
                (j - 1 < 0 || vwalls[i][j - 1] == 0) &&
                (hwalls[i][j] == 0)) {

                int connections = 0;
                if ((j + 1 >= board_size) ||
                    (hwalls[i][j + 1] == 1) ||
                    (i - 1 >= 0 && hwalls[i - 1][j + 1] == 1) ||
                    (i + 1 < board_size && hwalls[i + 1][j + 1] == 1) ||
                    (j + 2 < board_size && vwalls[i][j + 2] == 1))
                    connections += 1;
                if ((j - 1 < 0) ||
                    (hwalls[i][j - 1] == 1) ||
                    (i - 1 >= 0 && hwalls[i - 1][j - 1] == 1) ||
                    (i + 1 < board_size && hwalls[i + 1][j - 1] == 1) ||
                    (j - 2 >= 0 && vwalls[i][j - 2] == 1))
                    connections += 1;
                if ((i - 1 >= 0 && hwalls[i - 1][j] == 1) ||
                    (i + 1 < board_size && hwalls[i + 1][j] == 1))
                    connections += 1;

                if (connections < 2 ||
                    canPlaceWall(px, py, pgx, pgy, ox, oy, ogx, ogy, i, j, true, vwalls, hwalls)) {
                    vwall_actions[i][j] = 1;
                }
            }

            // Hwalls
            if ((hwalls[i][j] == 0 && old_legal_hwall_actions[i][j] == 1) &&
                (i + 1 >= board_size || hwalls[i + 1][j] == 0) &&
                (i - 1 < 0 || hwalls[i - 1][j] == 0) &&
                (vwalls[i][j] == 0)) {

                int connections = 0;
                if ((i + 1 >= board_size) ||
                    (vwalls[i + 1][j] == 1) ||
                    (j - 1 >= 0 && vwalls[i + 1][j - 1] == 1) ||
                    (j + 1 < board_size && vwalls[i + 1][j + 1] == 1) ||
                    (i + 2 < board_size && hwalls[i + 2][j] == 1))
                    connections += 1;

                if ((i - 1 < 0) ||
                    (vwalls[i - 1][j] == 1) ||
                    (j - 1 >= 0 && vwalls[i - 1][j - 1] == 1) ||
                    (j + 1 < board_size && vwalls[i - 1][j + 1] == 1) ||
                    (i - 2 >= 0 && hwalls[i - 2][j] == 1))
                    connections += 1;

                if ((j - 1 >= 0 && vwalls[i][j - 1] == 1) ||
                    (j + 1 < board_size && vwalls[i][j + 1] == 1))
                    connections += 1;

                if (connections < 2 ||
                    canPlaceWall(px, py, pgx, pgy, ox, oy, ogx, ogy, i, j, false, vwalls, hwalls)) {
                    hwall_actions[i][j] = 1;
                }
            }
        }
    }
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
getWallActions(int px, int py, int pgx, int pgy,
               int ox, int oy, int ogx, int ogy,
               std::vector<std::vector<int>> &vwalls,
               std::vector<std::vector<int>> &hwalls,
               int num_walls) {

    int board_size = (int) vwalls.size();
    std::vector<std::vector<int>> vwall_actions(board_size, std::vector<int>(board_size, 0));
    std::vector<std::vector<int>> hwall_actions(board_size, std::vector<int>(board_size, 0));
    if (num_walls > 0)
        setWallActions(px, py, pgx, pgy, ox, oy, ogx, ogy, vwalls, hwalls, vwall_actions, hwall_actions);

    return {vwall_actions, hwall_actions};
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
updateWallActions(int px, int py, int pgx, int pgy,
                  int ox, int oy, int ogx, int ogy,
                  const std::vector<std::vector<int>> &old_legal_vwall_actions,
                  const std::vector<std::vector<int>> &old_legal_hwall_actions,
                  std::vector<std::vector<int>> &vwalls,
                  std::vector<std::vector<int>> &hwalls,
                  int num_walls) {

    int board_size = (int) vwalls.size();
    std::vector<std::vector<int>> vwall_actions(board_size, std::vector<int>(board_size, 0));
    std::vector<std::vector<int>> hwall_actions(board_size, std::vector<int>(board_size, 0));
    if (num_walls > 0)
        updateWallActions_(px, py, pgx, pgy, ox, oy, ogx, ogy,
                           vwalls, hwalls,
                           old_legal_vwall_actions, old_legal_hwall_actions,
                           vwall_actions, hwall_actions);

    return {vwall_actions, hwall_actions};
}

std::vector<int> getValidActions(int px, int py, int pgx, int pgy,
                                 int ox, int oy, int ogx, int ogy,
                                 std::vector<std::vector<int>> &vwalls,
                                 std::vector<std::vector<int>> &hwalls,
                                 int num_walls) {
    int board_size = (int) vwalls.size();
    int action_size = 12 + 2 * board_size * board_size;
    std::vector<int> actions(action_size, 0);
    setPawnActions(px, py, ox, oy, vwalls, hwalls, actions);

    if (num_walls > 0) {
        std::vector<std::vector<int>> vwall_actions(board_size, std::vector<int>(board_size, 0));
        std::vector<std::vector<int>> hwall_actions(board_size, std::vector<int>(board_size, 0));
        setWallActions(px, py, pgx, pgy, ox, oy, ogx, ogy, vwalls, hwalls, vwall_actions, hwall_actions);

        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                int pawn_actions_shift = 12;
                actions[pawn_actions_shift + i * board_size + j] = vwall_actions[i][j];

                int vwall_actions_shift = pawn_actions_shift + (board_size * board_size);
                actions[vwall_actions_shift + i * board_size + j] = hwall_actions[i][j];
            }
        }
    }

    return actions;

}

inline PYBIND11_MODULE(QuoridorUtils, module) {
    module.doc() = "Quoridor Utils for engine V2";

    module.def("pathExists", &pathExists, "");
    module.def("findPath", &findPath, "");
    module.def("getPawnActions", &getPawnActions, "");
    module.def("getWallActions", &getWallActions, "");
    module.def("getValidActions", &getValidActions, "");
    module.def("updateWallActions", &updateWallActions, "");
}
