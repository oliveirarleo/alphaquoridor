#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "QuoridorMapSearchNode.h"
#include "MapInfo.h"
#include "QuoridorMapInfo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


inline bool PathExists(
        int start_x,
        int start_y,
        int end_x,
        int end_y,
        const std::vector<int> &world_map,
        int &map_width,
        int &map_height) {

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost
    // of travel across the terrain. Zero means the least possible difficulty
    // in travelling (think ice rink if you can skate) whilst 5 represents the
    // most difficult. 9 indicates that we cannot pass.

    // Create an instance of the search class...

    struct MapInfo Map;
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
        std::vector<int> &world_map,
        int &map_width,
        int &map_height) {

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost
    // of travel across the terrain. Zero means the least possible difficulty
    // in travelling (think ice rink if you can skate) whilst 5 represents the
    // most difficult. 9 indicates that we cannot pass.

    // Create an instance of the search class...

    struct MapInfo Map;
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

void getPawnActions(int player_x, int player_y, const std::vector<std::vector<int>> &walls,
                    const std::vector<std::vector<int>> &opponent_board, std::vector<int> &actions) {
    int board_size = (int) walls.size();
    //    NORTH
    //    Is there a wall blocking or is he in the edge?
    if ((player_y + 2 < board_size) && walls[player_y + 1][player_x] == 0) {
        // Is there a player on the north position?
        if (opponent_board[player_y + 2][player_x] == 1) {
            // JNE
            // Is there a wall on north of the opponent player or is he on the edge of the board?
            if (((player_y + 3 >= board_size) || walls[player_y + 3][player_x] == 1)) {
                // Is there a wall on the east of opponent?
                if ((player_x + 1 < board_size) && (walls[player_y + 2][player_x + 1] == 0)) {
                    // Jump NORTHEAST
                    actions[8] = 1;
                }
                // Is there a wall on the west of opponent?
                if ((player_x - 1 >= 0) && (walls[player_y + 2][player_x - 1] == 0)) {
                    // Jump NORTHWEST
                    actions[9] = 1;
                }
            }
                // JN
                // If there is a square and there is no player there
            else if ((player_y + 4 < board_size) && opponent_board[player_y + 4][player_x] != 1) {
                // Jump NORTH
                actions[4] = 1;
            }
        } else {
            // Move NORTH
            actions[0] = 1;
        }
    }

    //    SOUTH
    //    Is there a wall blocking or is he in the edge?
    if ((player_y - 2 >= 0) && walls[player_y - 1][player_x] == 0) {
        // Is there a player on the south position?
        if (opponent_board[player_y - 2][player_x] == 1) {
            // JSE
            // Is there a wall on south of the opponent player or is he on the edge of the board?
            if (((player_y - 3 < 0) || walls[player_y - 3][player_x] == 1)) {
                // Is there a wall on the east of opponent?
                if ((player_x + 1 < board_size) && (walls[player_y - 2][player_x + 1] == 0)) {
                    // Jump SOUTHEAST
                    actions[10] = 1;
                }
                // Is there a wall on the west of opponent?
                if ((player_x - 1 >= 0) && (walls[player_y - 2][player_x - 1] == 0)) {
                    // Jump SOUTHWEST
                    actions[11] = 1;
                }
            }
                // JS
                // If there is a square and there is no player there
            else if ((player_y - 4 > 0) && opponent_board[player_y - 4][player_x] != 1) {
                // Jump SOUTH
                actions[5] = 1;
            }
        } else {
            // Move SOUTH
            actions[1] = 1;
        }
    }


    //    EAST
    //    Is there a wall blocking or is he in the edge?
    if ((player_x + 2 < board_size) && walls[player_y][player_x + 1] == 0) {
        // Is there a player on the east position?
        if (opponent_board[player_y][player_x + 2] == 1) {
            // JEN
            // Is there a wall on east of the opponent player or is he on the edge of the board?
            if (((player_x + 3 >= board_size) || walls[player_y][player_x + 3] == 1)) {
                // Is there a wall on the north of opponent?
                if ((player_y + 1 < board_size) && (walls[player_y + 1][player_x + 2] == 0)) {
                    // Jump NORTHEAST
                    actions[8] = 1;
                }
                // Is there a wall on the south of opponent?
                if ((player_y - 1 >= 0) && (walls[player_y - 1][player_x + 2] == 0)) {
                    // Jump SOUTHEAST
                    actions[10] = 1;
                }
            }
                // JE
                // If there is a square and there is no player there
            else if ((player_x + 4 < board_size) && opponent_board[player_y][player_x + 4] != 1) {
                // Jump EAST
                actions[6] = 1;
            }
        } else {
            // Move EAST
            actions[2] = 1;
        }
    }


    //    WEST
    //    Is there a wall blocking or is he in the edge?
    if ((player_x - 2 >= 0) && walls[player_y][player_x - 1] == 0) {
        // Is there a player on the east position?
        if (opponent_board[player_y][player_x - 2] == 1) {
            // JEN
            // Is there a wall on east of the opponent player or is he on the edge of the board?
            if (((player_x - 3 < 0) || walls[player_y][player_x - 3] == 1)) {
                // Is there a wall on the north of opponent?
                if ((player_y + 1 < board_size) && (walls[player_y + 1][player_x - 2] == 0)) {
                    // Jump NORTHWEST
                    actions[9] = 1;
                }
                // Is there a wall on the south of opponent?
                if ((player_y - 1 >= 0) && (walls[player_y - 1][player_x - 2] == 0)) {
                    // Jump SOUTHWEST
                    actions[11] = 1;
                }
            }
                // JW
                // If there is a square and there is no player there
            else if ((player_y - 4 >= 0) && opponent_board[player_y][player_x - 4] != 1) {
                // Jump WEST
                actions[7] = 1;
            }
        } else {
            // Move WEST
            actions[3] = 1;
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
        walls[wall_y][wall_x] = 1;
        walls[wall_y + 1][wall_x] = 1;
        walls[wall_y - 1][wall_x] = 1;
    } else {
        walls[wall_y][wall_x] = 1;
        walls[wall_y][wall_x + 1] = 1;
        walls[wall_y][wall_x - 1] = 1;
    }
    std::vector<int> lin_walls(board_size * board_size);
    for (int i = 0; i < board_size; i++) {
        for (int j = 0; j < board_size; j++) {
            lin_walls[i * board_size + j] = walls[i][j];
        }
    }

    bool result = PathExists(player_x, player_y, player_end_x, player_end_y, lin_walls, board_size, board_size) &&
                  PathExists(opponent_x, opponent_y, opponent_end_x, opponent_end_y, lin_walls, board_size, board_size);

//    Remove wall
    if (is_vertical) {
        walls[wall_y][wall_x] = 0;
        walls[wall_y + 1][wall_x] = 0;
        walls[wall_y - 1][wall_x] = 0;
    } else {
        walls[wall_y][wall_x] = 0;
        walls[wall_y][wall_x + 1] = 0;
        walls[wall_y][wall_x - 1] = 0;
    }
    return result;
}


void getWallActions(std::vector<std::vector<int>> &walls,
                    int player_x, int player_y, int player_end_x, int player_end_y,
                    int opponent_x, int opponent_y, int opponent_end_x, int opponent_end_y,
                    int player_num_walls_placed, std::vector<int> &actions) {
    if (player_num_walls_placed >= 10)
        return;
    int board_size = (int) walls.size();
    int n = (board_size + 1) / 2 - 1;
    int pawn_actions = 12;
    int vwall_actions = pawn_actions + n * n;
    for (int i = 1; i < board_size; i += 2) {
        for (int j = 1; j < board_size; j += 2) {
            if (walls[i][j] == 0) {
                // Check vwall
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

                    if (connections < 2 ||
                        pathExistsForPlayers(walls, player_x, player_y, player_end_x, player_end_y, opponent_x,
                                             opponent_y, opponent_end_x, opponent_end_y, j, i, true)) {
                        actions[pawn_actions + j / 2 * n + i / 2] = 1;
                    }


                }
                // Check hwall
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

                    if (connections < 2 ||
                        pathExistsForPlayers(walls, player_x, player_y, player_end_x, player_end_y, opponent_x,
                                             opponent_y, opponent_end_x, opponent_end_y, j, i, false)) {
                        actions[vwall_actions + j / 2 * n + i / 2] = 1;
                    }

                }
            }

        }
    }

}


inline std::vector<int> GetValidMoves(
        std::vector<std::vector<std::vector<int>>> &board,
        int player) {

    int board_size = (int) board[0].size();
    int n = (board_size + 1) / 2;
    int action_size = 12 + 2 * ((n - 1) * (n - 1));
    std::vector<int> actions(action_size, 0);


    const int RED_POS = 2;
    const int RED_WALLS = 0;
    const int BLUE_POS = 3;
    const int BLUE_WALLS = 1;

// Choosing player
    int player_pos, player_walls, opponent_pos, player_goal_x, player_goal_y, opponent_goal_x, opponent_goal_y;
    if (player == 1) {
        player_pos = RED_POS;
        opponent_pos = BLUE_POS;
        player_walls = RED_WALLS;
        player_goal_x = (board_size - 1)/2;
        player_goal_y = board_size - 1;
        opponent_goal_x = (board_size - 1)/2;
        opponent_goal_y = 0;
    } else {
        player_pos = BLUE_POS;
        opponent_pos = RED_POS;
        player_walls = BLUE_WALLS;
        player_goal_x = (board_size - 1)/2;
        player_goal_y = 0;
        opponent_goal_x = (board_size - 1)/2;
        opponent_goal_y = board_size - 1;
    }

//    Unifying walls
    std::vector<std::vector<int>> walls(board_size);
    for (int i = 0; i < board_size; i++)
        walls[i].resize(board_size);

    joinWalls(board[RED_WALLS], board[BLUE_WALLS], walls);

//    Get Pawn actions
    int player_x = 0;
    int player_y = 0;
    int opponent_x = 0;
    int opponent_y = 0;
    findPlayerPosition(board[player_pos], player_x, player_y);
    findPlayerPosition(board[opponent_pos], opponent_x, opponent_y);
    getPawnActions(player_x, player_y, walls, board[opponent_pos], actions);

//    Get Wall actions
    int player_num_walls_placed = countWalls(board[player_walls]);
    getWallActions(walls,
                   player_x, player_y, player_goal_x, player_goal_y,
                   opponent_x, opponent_y, opponent_goal_x, opponent_goal_y,
                   player_num_walls_placed, actions);

    return actions;
}


inline PYBIND11_MODULE(QuoridorUtils, module) {
    module.doc() = "Python wrapper of AStar c++ implementation";

    module.def("PathExists", &PathExists, "Check if path exists");
    module.def("FindPath", &FindPath, "Find a collision-free path");
    module.def("GetValidMoves", &GetValidMoves, "Find a collision-free path");
}
