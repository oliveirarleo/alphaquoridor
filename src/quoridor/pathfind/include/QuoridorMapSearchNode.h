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
#include "MapInfo.h"
#include <limits>

const int INF = std::numeric_limits<int>::max();

class QuoridorMapSearchNode
{
public:
    int x;	 // the (x,y) positions of the node
    int y;	
    struct MapInfo map;

    QuoridorMapSearchNode() { x = 0; y = 0; }
    QuoridorMapSearchNode(int px, int py, const MapInfo &map_input) {
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
        (y == rhs.y) )
    {
        return true;
    }
    else
    {
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
    if (nodeGoal.x == 0 || nodeGoal.x == map.map_width-1)
    {
        return abs(x - nodeGoal.x);
    }

    return abs(y - nodeGoal.y);
}

bool QuoridorMapSearchNode::IsGoal( QuoridorMapSearchNode &nodeGoal )
{
    if ((nodeGoal.x == 0 || nodeGoal.x == map.map_width-1) && x == nodeGoal.x)
    {
        return true;
    } else if ((nodeGoal.y == 0 || nodeGoal.y == map.map_height-1) && y == nodeGoal.y)
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

    if( (GetMap( x-1, y ) < wall)
        && !((parent_x == x-2) && (parent_y == y))
        ) 
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x-2, y, map );
        astarsearch->AddSuccessor( NewNode );
    }	

    if( (GetMap( x, y-1 ) < wall)
        && !((parent_x == x) && (parent_y == y-2))
        ) 
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x, y-2, map );
        astarsearch->AddSuccessor( NewNode );
    }	

    if( (GetMap( x+1, y ) < wall)
        && !((parent_x == x+2) && (parent_y == y))
        ) 
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x+2, y, map );
        astarsearch->AddSuccessor( NewNode );
    }	

        
    if( (GetMap( x, y+1 ) < wall)
        && !((parent_x == x) && (parent_y == y+2))
        )
    {
        QuoridorMapSearchNode NewNode = QuoridorMapSearchNode( x, y+2, map );
        astarsearch->AddSuccessor( NewNode );
    }	

    return true;
}

// given this node, what does it cost to move to successor. In the case
// of our map the answer is the map terrain value at this node since that is 
// conceptually where we're moving

float QuoridorMapSearchNode::GetCost( QuoridorMapSearchNode &successor )
{
    int next_node = GetMap( successor.x, successor.y );
    if (next_node >= 1)
        return INF;
    return 1.0;
}


inline int QuoridorMapSearchNode::GetMap(int x, int y)
{
    if( x < 0 ||
        x >= map.map_width ||
            y < 0 ||
            y >= map.map_height
        )
    {
        return INF;
    }

    return map.world_map[(y*map.map_width)+x];
}


#endif