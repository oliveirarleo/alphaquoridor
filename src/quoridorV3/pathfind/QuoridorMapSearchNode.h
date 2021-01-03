////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// STL A* Search implementation
// (C)2001 Justin Heyes-Jones
//
// Finding a path on a simple grid maze
// This shows how to do shortest path finding using A*

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef QuoridorMapSearchNode_H
#define QuoridorMapSearchNode_H

#include <iostream>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "QuoridorMapInfo.h"
#include <limits>

const int INF = std::numeric_limits<int>::max();


// If is blocked North of player
inline bool hasHWN(const std::vector<std::vector<int>> &hwalls, int x, int y) {
    int board_size = (int)hwalls.size();
    return (y >= board_size) || (y<0) || (x < board_size && hwalls[x][y] == 1) || (x > 0 && hwalls[x-1][y] == 1);
}

// If is blocked East of player
inline bool hasVWE(const std::vector<std::vector<int>> &vwalls, int x, int y) {
    int board_size = (int)vwalls.size();
    return (x >= board_size) || (x<0) || (y < board_size && vwalls[x][y] == 1) || (y > 0 && vwalls[x][y-1] == 1);
}

class QuoridorMapSearchNode
{
public:
    int x;	 // the (x,y) positions of the node
    int y;	
    struct QuoridorMapInfo map;

    QuoridorMapSearchNode() { x = 0; y = 0; }
    QuoridorMapSearchNode(int px, int py, const QuoridorMapInfo &map_input) {
        x = px;
        y = py;
        map = map_input;
    }

    float GoalDistanceEstimate( QuoridorMapSearchNode &nodeGoal );
    bool IsGoal( QuoridorMapSearchNode &nodeGoal );
    bool GetSuccessors( AStarSearch<QuoridorMapSearchNode> *astarsearch, QuoridorMapSearchNode *parent_node );
    float GetCost( QuoridorMapSearchNode &successor );
    bool IsSameState( QuoridorMapSearchNode &rhs );

    void PrintNodeInfo(); 

    inline int GetMap(int x, int y);

};

bool QuoridorMapSearchNode::IsSameState( QuoridorMapSearchNode &rhs )
{

    // same state in a maze search is simply when (x,y) are the same
    if( (x == rhs.x) &&
        (y == rhs.y) ) {
        return true;
    }
    else {
        return false;
    }

}

void QuoridorMapSearchNode::PrintNodeInfo()
{
    char str[100];
    sprintf( str, "Node position : (%d,%d)\n", x,y );

    std::cout << str;
}

// Here's the heuristic function that estimates the distance from a Node
// to the Goal. 

float QuoridorMapSearchNode::GoalDistanceEstimate( QuoridorMapSearchNode &nodeGoal )
{
    return abs(y - nodeGoal.y);
}

bool QuoridorMapSearchNode::IsGoal( QuoridorMapSearchNode &nodeGoal )
{
    if (y == nodeGoal.y)
    {
        return true;
    }
    return false;
}

// This generates the successors to the given Node. It uses a helper function called
// AddSuccessor to give the successors to the AStar class. The A* specific initialisation
// is done for each node internally, so here you just set the state information that
// is specific to the application
bool QuoridorMapSearchNode::GetSuccessors( AStarSearch<QuoridorMapSearchNode> *astarsearch, QuoridorMapSearchNode *parent_node )
{
    const int wall = 1;
    int parent_x = -1; 
    int parent_y = -1; 

    if( parent_node )
    {
        parent_x = parent_node->x;
        parent_y = parent_node->y;
    }


    // QuoridorMapSearchNode NewNode;

    // push each possible move except allowing the search to go backwards
    // NORTH
    if(!hasHWN(map.hwalls, x, y)
        && !((parent_x == x) && (parent_y == y+1)))
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x, y+1, map );
        astarsearch->AddSuccessor( NewNode );
    }

    // SOUTH
    if( !hasHWN(map.hwalls, x, y-1)
        && !((parent_x == x) && (parent_y == y-1)))
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x, y-1, map );
        astarsearch->AddSuccessor( NewNode );
    }

    // EAST
    if( !hasVWE(map.vwalls, x, y)
        && !((parent_x == x+1) && (parent_y == y)))
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x+1, y, map );
        astarsearch->AddSuccessor( NewNode );
    }	
    // WEST
    if( !hasVWE(map.vwalls, x-1, y)
        && !((parent_x == x-1) && (parent_y == y)))
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode(x-1, y, map);
        astarsearch->AddSuccessor(NewNode);
    }
    return true;
}

// given this node, what does it cost to move to successor. In the case
// of our map the answer is the map terrain value at this node since that is 
// conceptually where we're moving

float QuoridorMapSearchNode::GetCost( QuoridorMapSearchNode &successor )
{
//    int next_node = GetMap( successor.x, successor.y );
//    if (next_node >= 1)
//        return INF;
    return 1.0;
}


//inline int QuoridorMapSearchNode::GetMap(int x, int y)
//{
//    if( x < 0 ||
//        x >= map.map_width ||
//            y < 0 ||
//            y >= map.map_height
//        )
//    {
//        return INF;
//    }
//
//    return map.world_map[x][y];
//}


#endif