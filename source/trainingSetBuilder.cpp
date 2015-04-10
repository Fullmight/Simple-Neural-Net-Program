/*
Author: Cameron. A. Pickerill-Trinitapoli
Date: 15-03-2015
Version: 1.5

This is a simple script to generate training
data for a simple neural network.
The values after "topology: " refer to the number
of neurons in each layer of the network (3 layer net)
initially these are 2 6 2
first: Input layer, Second: Hidden layer, Third: Output
layer.
You can replace these with any numbers you wish, as long
as the program is also modified to generate the correct
number of inputs and outputs so that an error will not occur.
Valid input and output ranges are between -1 and 1. Net
output will be in double form so that if you wish
to have a gradient response that is possible (i.e. 
the closer to the target, the higher the confidence
of the network is).

Example training file format:
topology: (int, input nodes) (int, hidden nodes) (int, output nodes)
in: (int) (int)
out: (int) (int)

//Ex2.

topology: 2 6 2
in: 0.0 -1.0 
out: 1.0 0.0


Use instructions:
The simplest method of use
is to modify what the target val
is for a given val of n1 or n2.
alternatively you may change the
random value creation to be anything
between -1 and 1 and create appropriate
target values of your own to correspond
to particular inputs.

A externally generated file
may be used for any values
inside of the listed ranges
as long as it follows the format
given above.
*/


#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main()
{
    //build random training sets
    cout << "topology: 2 6 2" << endl;
    for(int iterator = 2000; iterator >=0; --iterator) {
        int n1 = (int)(rand() % 2 - 1);//generates a random value
        int n2 = (int)(rand() % 2 - 1);//from -1 to 1.
        int target1 = 0; // either 0 or 1
        int target2 = 0; // Either -1, 0, or  1.
        //if range is medium/close move to attack
        //else ignore for target 1
        //For target 2, take action
        //appropriate to player action
        //represented by -1, 0, or 1.
        //examples in write up with network graph.
        if (n1 >= 0) {
            target1 = 1;
        }
        else {
            target1 = 0;
        };


        //If target is blocking, net should ignore them
        //if target is attacking, net should attack back
        //if target is using ranged, net should block.
         if (n2 == -1) {
            target2 = 0;
        }
        else if (n2 == 0) {
            target2 = 1;
        }
        else {
            target2 = -1;
        }
        cout << "in: " << n1 << ".0 " << n2 << ".0 " << endl;
        cout << "out: " << target1 << ".0" << " " 
            << target2 << ".0" << endl;
        //any input in this format of valid ranges will be accepted
        //in this way you may use some pre-generated training data
        //or data extracted from your primary program.
    }
}
