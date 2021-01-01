#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "QuoridorMapSearchNode.h"
#include "QuoridorMapInfo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


inline bool PathExists(
        int start_x,
        int start_y,
        int end_x,
        int end_y,
        const std::vector<std::vector<int>> &world_map,
        int &map_width,
        int &map_height) {

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost
    // of travel across the terrain. Zero means the least possible difficulty
    // in travelling (think ice rink if you can skate) whilst 5 represents the
    // most difficult. 9 indicates that we cannot pass.

    // Create an instance of the search class...

    struct QuoridorMapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

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

inline std::tuple<std::vector<int>, int> FindPath(
        std::vector<int> &start,
        std::vector<int> &end,
        std::vector<std::vector<int>> &world_map,
        int &map_width,
        int &map_height) {

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost
    // of travel across the terrain. Zero means the least possible difficulty
    // in travelling (think ice rink if you can skate) whilst 5 represents the
    // most difficult. 9 indicates that we cannot pass.

    // Create an instance of the search class...

    struct QuoridorMapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

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

            // std::cout << "Solution steps " << steps << endl;

            // Once you're done with the solution you can free the nodes up
            astarsearch.FreeSolutionNodes();

        } else if (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_FAILED) {
            std::cout << "Search terminated. Did not find goal state\n";
        }

        // Display the number of loops the search went through
        // std::cout << "SearchSteps : " << SearchSteps << "\n";

        SearchCount++;

        astarsearch.EnsureMemoryFreed();

    }
    return {path_full, steps};
}


void printBoard(const std::vector<std::vector<int>> &board) {
    for (const auto &line : board) {
        for (const auto &e : line) {
            std::cout << e;
        }
        std::cout << '\n';
    }
    std::cout << '\n';

}

int countWalls(const std::vector<std::vector<int>> &walls) {
    int num_walls = 0;
    int board_size = (int) walls.size();
    for (int i = 1; i < board_size; i += 2) {
        for (int j = 1; j < board_size; j += 2) {
            num_walls += walls[i][j];
        }
    }
    return num_walls;
}

void findPlayerPosition(const std::vector<std::vector<int>> &board, int &x, int &y) {
    int board_size = (int) board.size();
    for (int i = 0; i < board_size; i += 2) {
        for (int j = 0; j < board_size; j += 2) {
            if (board[i][j] == 1) {
                x = j;
                y = i;
                return;
            }
        }
    }
}

void joinWalls(const std::vector<std::vector<int>> &red_walls, const std::vector<std::vector<int>> &blue_walls,
               std::vector<std::vector<int>> &walls) {

    int board_size = (int) red_walls.size();

    for (int i = 0; i < board_size; i += 2) {
        for (int j = 1; j < board_size; j += 2) {
            if (red_walls[i][j] == 1 || blue_walls[i][j] == 1) {
                walls[i][j] = 1;
            } else {
                walls[i][j] = 0;
            }
        }
    }

    for (int i = 1; i < board_size; i += 2) {
        for (int j = 0; j < board_size; j++) {
            if (red_walls[i][j] == 1 || blue_walls[i][j] == 1) {
                walls[i][j] = 1;
            } else {
                walls[i][j] = 0;
            }
        }
    }

}

void getPawnActions(int player_x, int player_y, int opponent_x, int opponent_y,
                    const std::vector<std::vector<int>> &walls, std::vector<int> &actions) {
    int N = 0;
    int S = 1;
    int E = 2;
    int W = 3;
    int JN = 4;
    int JS = 5;
    int JE = 6;
    int JW = 7;
    int NE = 8;
    int NW = 9;
    int SE = 10;
    int SW = 11;

    int board_size = (int) walls.size();
    //    NORTH
    //    Is there a wall blocking or is he in the edge?
    if ((player_y + 2 < board_size) && walls[player_x][player_y + 1] == 0) {
        // Is there a player on the north position?
        if (player_x == opponent_x && (player_y + 2) == opponent_y) {
            // JNE
            // Is there a wall on north of the opponent player or is he on the edge of the board?
            if (((player_y + 3 >= board_size) || walls[player_x][player_y + 3] == 1)) {
                // Is there a wall on the east of opponent?
                if ((player_x + 1 < board_size) && (walls[player_x + 1][player_y + 2] == 0)) {
                    // Jump NORTHEAST
                    actions[NE] = 1;
                }
                // Is there a wall on the west of opponent?
                if ((player_x - 1 >= 0) && (walls[player_x - 1][player_y + 2] == 0)) {
                    // Jump NORTHWEST
                    actions[NW] = 1;
                }
            }
                // JN
                // If there is a square and there is no player there
            else if (player_y + 4 < board_size) {
                // Jump NORTH
                actions[JN] = 1;
            }
        } else {
            // Move NORTH
            actions[N] = 1;
        }
    }

    //    SOUTH
    //    Is there a wall blocking or is he in the edge?
    if ((player_y - 2 >= 0) && walls[player_x][player_y - 1] == 0) {
        // Is there a player on the south position?
        if ((player_x == opponent_x && (player_y - 2) == opponent_y)) {
            // JSE
            // Is there a wall on south of the opponent player or is he on the edge of the board?
            if (((player_y - 3 < 0) || walls[player_x][player_y - 3] == 1)) {
                // Is there a wall on the east of opponent?
                if ((player_x + 1 < board_size) && (walls[player_x + 1][player_y - 2] == 0)) {
                    // Jump SOUTHEAST
                    actions[SE] = 1;
                }
                // Is there a wall on the west of opponent?
                if ((player_x - 1 >= 0) && (walls[player_x - 1][player_y - 2] == 0)) {
                    // Jump SOUTHWEST
                    actions[SW] = 1;
                }
            }
                // JS
                // If there is a square and there is no player there
            else if (player_y - 4 >= 0) {
                // Jump SOUTH
                actions[JS] = 1;
            }
        } else {
            // Move SOUTH
            actions[S] = 1;
        }
    }


    //    EAST
    //    Is there a wall blocking or is he in the edge?
    if ((player_x + 2 < board_size) && walls[player_x + 1][player_y] == 0) {
        // Is there a player on the east position?
        if ((player_x + 2) == opponent_x && player_y == opponent_y) {
            // JEN
            // Is there a wall on east of the opponent player or is he on the edge of the board?
            if (((player_x + 3 >= board_size) || walls[player_x + 3][player_y] == 1)) {
                // Is there a wall on the north of opponent?
                if ((player_y + 1 < board_size) && (walls[player_x + 2][player_y + 1] == 0)) {
                    // Jump NORTHEAST
                    actions[NE] = 1;
                }
                // Is there a wall on the south of opponent?
                if ((player_y - 1 >= 0) && (walls[player_x + 2][player_y - 1] == 0)) {
                    // Jump SOUTHEAST
                    actions[SE] = 1;
                }
            }
                // JE
                // If there is a square and there is no player there
            else if (player_x + 4 < board_size) {
                // Jump EAST
                actions[JE] = 1;
            }
        } else {
            // Move EAST
            actions[E] = 1;
        }
    }

    //    WEST
    //    Is there a wall blocking or is he in the edge?
    if ((player_x - 2 >= 0) && walls[player_x - 1][player_y] == 0) {
        // Is there a player on the east position?
        if ((player_x - 2) == opponent_x && player_y == opponent_y) {
            // JEN
            // Is there a wall on east of the opponent player or is he on the edge of the board?
            if (((player_x - 3 < 0) || walls[player_x - 3][player_y] == 1)) {
                // Is there a wall on the north of opponent?
                if ((player_y + 1 < board_size) && (walls[player_x - 2][player_y + 1] == 0)) {
                    // Jump NORTHWEST
                    actions[NW] = 1;
                }
                // Is there a wall on the south of opponent?
                if ((player_y - 1 >= 0) && (walls[player_x - 2][player_y - 1] == 0)) {
                    // Jump SOUTHWEST
                    actions[SW] = 1;
                }
            }
                // JW
                // If there is a square
            else if (player_x - 4 >= 0) {
                // Jump WEST
                actions[JW] = 1;
            }
        } else {
            // Move WEST
            actions[W] = 1;
        }
    }
}

bool pathExistsForPlayers(std::vector<std::vector<int>> &walls,
                          int player_x, int player_y, int player_end_x, int player_end_y,
                          int opponent_x, int opponent_y, int opponent_end_x, int opponent_end_y,
                          int wall_x, int wall_y, bool is_vertical) {
    int board_size = (int) walls.size();

//    Insert wall
    if (is_vertical) {
        walls[wall_x][wall_y] = 1;
        walls[wall_x][wall_y + 1] = 1;
        walls[wall_x][wall_y - 1] = 1;
    } else {
        walls[wall_x][wall_y] = 1;
        walls[wall_x + 1][wall_y] = 1;
        walls[wall_x - 1][wall_y] = 1;
    }

    bool result = PathExists(player_x, player_y, player_end_x, player_end_y, walls, board_size, board_size) &&
                  PathExists(opponent_x, opponent_y, opponent_end_x, opponent_end_y, walls, board_size, board_size);

//    Remove wall
    if (is_vertical) {
        walls[wall_x][wall_y] = 0;
        walls[wall_x][wall_y + 1] = 0;
        walls[wall_x][wall_y - 1] = 0;
    } else {
        walls[wall_x][wall_y] = 0;
        walls[wall_x + 1][wall_y] = 0;
        walls[wall_x - 1][wall_y] = 0;
    }
    return result;
}


void getWallActions(std::vector<std::vector<int>> &walls,
                    int player_x, int player_y, int player_end_x, int player_end_y,
                    int opponent_x, int opponent_y, int opponent_end_x, int opponent_end_y, std::vector<int> &actions,
                    int num_walls) {
    if (num_walls <= 0)
        return;
    int board_size = (int) walls.size();
    int n = (board_size + 1) / 2 - 1;
    int pawn_actions = 12;
    int vwall_actions = pawn_actions + n * n;
    for (int i = 1; i < board_size; i += 2) {
        for (int j = 1; j < board_size; j += 2) {
            if (walls[i][j] == 0) {
                // Check vwall
                if ((walls[i][j + 1] == 0) && (walls[i][j - 1] == 0)) {

                    int connections = 0;
                    if ((j + 3 >= board_size) || (walls[i][j + 3] == 1) || (walls[i + 1][j + 2] == 1) ||
                        (walls[i - 1][j + 2] == 1))
                        connections += 1;
                    if ((j - 3 < 0) || (walls[i][j - 3] == 1) || (walls[i + 1][j - 2] == 1) ||
                        (walls[i - 1][j - 2] == 1))
                        connections += 1;
                    if ((walls[i + 1][j] == 1) || (walls[i - 1][j] == 1))
                        connections += 1;

                    if (connections < 2 || pathExistsForPlayers(walls, player_x, player_y, player_end_x, player_end_y,
                                                                opponent_x, opponent_y, opponent_end_x, opponent_end_y,
                                                                i, j, true)) {
//                        if (connections < 2){
                        actions[pawn_actions + i / 2 * n + j / 2] = 1;
                    }


                }
                // Check hwall
                if ((walls[i + 1][j] == 0) && (walls[i - 1][j] == 0)) {
                    int connections = 0;
                    if ((i + 3 >= board_size) || (walls[i + 3][j] == 1) || (walls[i + 2][j + 1] == 1) ||
                        (walls[i + 2][j - 1] == 1))
                        connections += 1;
                    if ((i - 3 < 0) || (walls[i - 3][j] == 1) || (walls[i - 2][j + 1] == 1) ||
                        (walls[i - 2][j - 1] == 1))
                        connections += 1;
                    if ((walls[i][j + 1] == 1) || (walls[i][j - 1] == 1))
                        connections += 1;

                    if (connections < 2 || pathExistsForPlayers(walls, player_x, player_y, player_end_x, player_end_y,
                                                                opponent_x, opponent_y, opponent_end_x, opponent_end_y,
                                                                i, j, false)) {
//                    if (connections < 2){
                        actions[vwall_actions + i / 2 * n + j / 2] = 1;
                    }

                }
            }

        }
    }
}

std::vector<std::vector<std::vector<int>>> getWallActions2(std::vector<std::vector<int>> &walls,
                                                           int player_x, int player_y, int player_end_x,
                                                           int player_end_y,
                                                           int opponent_x, int opponent_y, int opponent_end_x,
                                                           int opponent_end_y, int num_walls) {

    int board_size = (int) walls.size();
    int n = (board_size + 1) / 2 - 1;
    std::vector<std::vector<std::vector<int>>> actions(n,
                                                       std::vector<std::vector<int>>(n,
                                                                                     std::vector<int>(n, 0)));
    if (num_walls <= 0)
        return actions;

    int vwall_idx = 0;
    int hwall_idx = 1;
    for (int i = 1; i < board_size; i += 2) {
        for (int j = 1; j < board_size; j += 2) {
            if (walls[i][j] == 0) {
                // Check vwall
                if ((walls[i][j + 1] == 0) && (walls[i][j - 1] == 0)) {

                    int connections = 0;
                    if ((j + 3 >= board_size) || (walls[i][j + 3] == 1) || (walls[i + 1][j + 2] == 1) ||
                        (walls[i - 1][j + 2] == 1))
                        connections += 1;
                    if ((j - 3 < 0) || (walls[i][j - 3] == 1) || (walls[i + 1][j - 2] == 1) ||
                        (walls[i - 1][j - 2] == 1))
                        connections += 1;
                    if ((walls[i + 1][j] == 1) || (walls[i - 1][j] == 1))
                        connections += 1;

                    if (connections < 2 || pathExistsForPlayers(walls, player_x, player_y, player_end_x, player_end_y,
                                                                opponent_x, opponent_y, opponent_end_x, opponent_end_y,
                                                                i, j, true)) {
                        actions[vwall_idx][i/2][j/2] = 1;
                    }
                }
                // Check hwall
                if ((walls[i + 1][j] == 0) && (walls[i - 1][j] == 0)) {
                    int connections = 0;
                    if ((i + 3 >= board_size) || (walls[i + 3][j] == 1) || (walls[i + 2][j + 1] == 1) ||
                        (walls[i + 2][j - 1] == 1))
                        connections += 1;
                    if ((i - 3 < 0) || (walls[i - 3][j] == 1) || (walls[i - 2][j + 1] == 1) ||
                        (walls[i - 2][j - 1] == 1))
                        connections += 1;
                    if ((walls[i][j + 1] == 1) || (walls[i][j - 1] == 1))
                        connections += 1;

                    if (connections < 2 || pathExistsForPlayers(walls, player_x, player_y, player_end_x, player_end_y,
                                                                opponent_x, opponent_y, opponent_end_x, opponent_end_y,
                                                                i, j, false)) {
                        actions[hwall_idx][i/2][j/2] = 1;
                    }
                }
            }
        }
    }
    return actions;
}


inline std::vector<int> GetValidPawnActions(int player_x, int player_y,
                                            int opponent_x, int opponent_y,
                                            std::vector<std::vector<int>> &walls) {
    std::vector<int> actions(12, 0);
//    Get Pawn actions
    getPawnActions(player_x, player_y, opponent_x, opponent_y, walls, actions);
    return actions;
}


inline std::vector<int> GetValidActions(
        int player_x, int player_y, int player_end_x, int player_end_y,
        int opponent_x, int opponent_y, int opponent_end_x, int opponent_end_y,
        std::vector<std::vector<int>> &walls, int num_walls) {

    int board_size = (int) walls[0].size();
    int n = (board_size + 1) / 2;
    int action_size = 12 + 2 * ((n - 1) * (n - 1));
    std::vector<int> actions(action_size, 0);

//    Get Pawn actions
    getPawnActions(player_x, player_y, opponent_x, opponent_y, walls, actions);

//    Get Wall actions
    getWallActions(walls,
                   player_x, player_y, player_end_x, player_end_y,
                   opponent_x, opponent_y, opponent_end_x, opponent_end_y, actions, num_walls);

    return actions;
}


inline PYBIND11_MODULE(QuoridorUtils, module) {
    module.doc() = "Python wrapper of AStar c++ implementation";

    module.def("PathExists", &PathExists, "Check if path exists");
    module.def("pathExistsForPlayers", &pathExistsForPlayers, "Check if path exists");
    module.def("FindPath", &FindPath, "Find a collision-free path");
    module.def("GetValidActions", &GetValidActions, "");
    module.def("getWallActions2", &getWallActions2, "");
    module.def("GetValidPawnActions", &GetValidPawnActions, "");
}
