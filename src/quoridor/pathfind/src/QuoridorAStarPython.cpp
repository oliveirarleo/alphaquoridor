#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "QuoridorMapSearchNode.h"
#include "MapInfo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "get_combination.h"
#include "find_path.h"


inline bool PathExists(
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

    // QuoridorMapSearchNode nodeStart;
    QuoridorMapSearchNode nodeStart = QuoridorMapSearchNode(start[0], start[1], Map);
    QuoridorMapSearchNode nodeEnd(end[0], end[1], Map);

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

inline bool PathExistsAll(
        std::vector<int> &start,
        std::vector<int> &end,
        std::vector<int> &world_map,
        int &map_width,
        int &map_height) {
    struct MapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;
    int num_search = start.size() / 2;
    AStarSearch<QuoridorMapSearchNode> astarsearch;

    for (int s = 0; s < num_search; s++) {
        // QuoridorMapSearchNode nodeStart;
        QuoridorMapSearchNode nodeStart = QuoridorMapSearchNode(start[0 + 2 * s], start[1 + 2 * s], Map);
        QuoridorMapSearchNode nodeEnd(end[0 + 2 * s], end[1 + 2 * s], Map);
        std::cout << "start: " << start[0 + 2 * s] << " " << start[1 + 2 * s] << " end: " << end[0 + 2 * s] << " "
                  << end[1 + 2 * s] << "\n";
        // Set Start and goal states
        astarsearch.SetStartAndGoalStates(nodeStart, nodeEnd);

        unsigned int SearchState;
        do {
            SearchState = astarsearch.SearchStep();
        } while (SearchState == AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_SEARCHING);

        if (SearchState != AStarSearch<QuoridorMapSearchNode>::SEARCH_STATE_SUCCEEDED) {
            astarsearch.EnsureMemoryFreed();
            return false;
        }
        astarsearch.FreeSolutionNodes();
        astarsearch.EnsureMemoryFreed();
    }
    return true;
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


inline std::tuple<std::vector<std::vector<int>>, std::vector<int>> FindPathAll(
        std::vector<int> agent_position,
        std::vector<int> targets_position,
        std::vector<int> &world_map,
        int &map_width,
        int &map_height) {
    struct MapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

    int num_targets = targets_position.size() / 2;
    std::vector<int> start_goal_pair = get_combination(num_targets + 1, 2);
    std::vector<std::vector<int>> path_all;
    std::vector<int> steps_all;
    int start[2];
    int goal[2];

    for (unsigned long idx = 0; idx < start_goal_pair.size(); idx = idx + 2) {
        int start_idx = start_goal_pair[idx];
        int goal_idx = start_goal_pair[idx + 1];

        if (start_idx != 0) {
            start[0] = targets_position[2 * (start_idx - 1)];
            start[1] = targets_position[2 * (start_idx - 1) + 1];
        } else {
            start[0] = agent_position[0];
            start[1] = agent_position[1];
        }

        if (goal_idx != 0) {
            goal[0] = targets_position[2 * (goal_idx - 1)];
            goal[1] = targets_position[2 * (goal_idx - 1) + 1];

        } else {
            goal[0] = agent_position[0];
            goal[1] = agent_position[1];
        }
        auto[path_short_single, steps_used] = find_path(start, goal, Map);
        path_all.push_back(path_short_single);
        steps_all.push_back(steps_used);
    }

    // return path_all;
    return {path_all, steps_all};
}

inline std::vector<int> GetValidMoves(
        std::vector<std::vector<std::vector<int>>> &board,
        int player) {
    int n = ((int)board[0].size() + 1) / 2;
    int action_size = 12 + 2 * ((n - 1) * (n - 1));
    std::vector<int> actions(action_size, 0);

    for (const auto& boardtype : board) {
        for (const auto& line : boardtype) {
            for (const auto& e : line) {
                std::cout << e;
            }
            std::cout << '\n';
        }

        std::cout << '\n';
    }

    return actions;
}


inline PYBIND11_MODULE(QuoridorAStarPython, module) {
    module.doc() = "Python wrapper of AStar c++ implementation";

    module.def("PathExists", &PathExists, "Check if path exists");
    module.def("PathExistsAll", &PathExists, "Check if path exists");
    module.def("FindPath", &FindPath, "Find a collision-free path");
    module.def("FindPathAll", &FindPathAll, "Find a collision-free path");
    module.def("GetValidMoves", &GetValidMoves, "Find a collision-free path");
}
