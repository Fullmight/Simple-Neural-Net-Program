/*
Author: Cameron Pickerill-Trinitapoli
Class: CS251
ID: G03573349
Reference Material: http://en.wikipedia.org/wiki/Artificial_neural_network,
http://www.ai-junkie.com/ann/evolved/nnt1.html, http://tinyurl.com/mfxhctn,
https://vimeo.com/19569529, https://vimeo.com/56882963, & Epp.
Date: 03-15-2015
Version 1.1

This is a very simple neural network; an example of a directed
graph application in computer science. This program creates
a weighted graph with 3 layers (input, hidden, output) and
then trains the graph using gradient descent in order to 
search out a solution to a given problem by tweaking edge
weights between neurons. In this case, the goal is for the
program to produce a Neural Net that controls a hypothetical
non-player character in a hypothetical game. The Net must
instruct this non-player character as to what it should
do in response to actions by the player. The inputs
and corresponding desired outputs are outlined in the
project writeup in the Neural Net diagram.

This method can be used for a variety of situations.
Essentially any scenario where you have some number of
inputs, some number of outputs, and an unknown function
to preform on the data to get the correct outputs each
time is a valid problem space.

Some definition notes:
Topology refers to its use in neural networks and computer science
to describe a "Network Topology"; an arrangement of
the various elements of a computer network (links/nodes).
Delta refers to the "Delta Rule," which is a gradient descent
learning rule for updating the weights of the inputs to artificial
neurons.

Smoothing Factor refers to the size of the deviation when training
the network. Aka how "spiky" the functions are.

Error Rate refers to the deviation from the goal output.



*/


//includes
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>


using namespace std;

// ************** Class Learn ********** //
//Fetches and processes data.
//Nothing really special here, just setup
//for feeding training data through the net.
//Gets the file name, iterates and parses
//data, checks for eof. I'd have rather
//added the training data generation to
//the program with inputs for what you
//want to train but ran out of time for
//extras.
class Learn
{
public:
    Learn(const string filename);
    bool eofCheck(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input vals and output vals from the file.
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOuts(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

//gets the number of neurons in each layer.
void Learn::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream streamString(line);
    streamString >> label;
    if (this->eofCheck() || label.compare("topology:") != 0) {
        abort();
    }

    while (!streamString.eof()) {
        unsigned tempNum;
        streamString >> tempNum;
        topology.push_back(tempNum);
    }

    return;
}

Learn::Learn(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned Learn::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream streamLine(line);

    string label;
    streamLine>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (streamLine >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

//Fetches our target output value that the
//net will attempt to match.
unsigned Learn::getTargetOuts(vector<double> &targetOutVals)
{
    targetOutVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream streamLine(line);

    string label;
    streamLine>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (streamLine >> oneValue) {
            targetOutVals.push_back(oneValue);
        }
    }

    return targetOutVals.size();
}

//Helps with modeling the weights
//placed on neuron connections (edge weights).
struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;


// ************** Class Neuron ******* //
//This contains weights from this neuron
//to other neuron's this neruon feeds.
class Neuron {
public: 
    Neuron(unsigned numberOfOuts, unsigned index);
    void setOutVal(double val){ m_outVal = val; }
    double getOutVal(void) const { return m_outVal; }
    void feedForward(const Layer &priorLayer);
    void outGradients(double targetVal); //calculates the output gradients.
    void hiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &priorLayer);
private:
    static double eta;
    static double alpha;
    static double activationFunction(double x);
    static double activationFunctionDerivative(double x);
    static double randomWeight(void){ return rand() / double(RAND_MAX); }
    double m_outVal;
    double sumDerivOfWeights(const Layer &nextLayer) const;
    vector<Connection> m_outWeights;
    unsigned m_index;
    double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

// Updates weights in the connection container.
// from neurons of the previous layer.
void Neuron::updateInputWeights(Layer &priorLayer){

    for(unsigned nWeightLoop = 0; nWeightLoop < priorLayer.size(); ++nWeightLoop){
        Neuron &neuron = priorLayer[nWeightLoop];
        double prevDeltaWeight = neuron.m_outWeights[m_index].deltaWeight;
        
        //change in weight calculation
        //along a connection.
        double newDeltaWeight =
                eta//learning rate
                * neuron.getOutVal()
                * m_gradient
                + alpha //"momentum" keeps going in same direction.
                * prevDeltaWeight;
        neuron.m_outWeights[m_index].deltaWeight = newDeltaWeight;
        neuron.m_outWeights[m_index].weight += newDeltaWeight;
    }
}

//Collects the contribution to errors at nodes that are fed to.
//This is actually pretty hard for me to understand but
//essentially it's another tool for figuring out how many
//errors there are that we use to directionally improve
//the preformance of the network.
double Neuron::sumDerivOfWeights(const Layer &nextLayer) const {
    double sum = 0.0;
    
    for(unsigned nSumLoop = 0; nSumLoop < nextLayer.size() - 1; ++nSumLoop){
        sum += m_outWeights[nSumLoop].weight * nextLayer[nSumLoop].m_graLdient;
    }
    
    return sum;
}

//A technique with this kind of neural net involves
//taking the sum of the derivatives of the next layer
//this has much the same purpose as the outGradients
//function; just a very different calculation.
//DOW/derivOfWeights = derivative of weights.
void Neuron::hiddenGradients(const Layer &nextLayer){
    double dow = sumDerivOfWeights(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outVal);
}

//This keeps things headed in the direction of slowly reducing overall
//Neural Net Error.
void Neuron::outGradients(double targetVal){
    double delta = targetVal - m_outVal;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outVal);
}

double Neuron::activationFunction(double x){
    //For this practice program we are going
    //to use a hyperbolic tangent function with
    //an output range from [-1.0] to [1.0].
    //this gives us a nice but steep curve
    //rather than a harsh step.
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x){
    //This is essentially a quick way to get
    //the derivative of the tanh(x) function
    //that I had recommended to me. To the best
    //of my understanding of neural nets it
    //could be replaced by the derivative of
    //whatever function suits your needs for
    //the data you are computing.
    return 1.0 - (x * x);
}

//forward directed data.
void Neuron::feedForward(const Layer &priorLayer)
{
    double sum = 0.0;
    
    for(unsigned priorNeuron = 0; priorNeuron < priorLayer.size(); ++priorNeuron) {
        sum += priorLayer[priorNeuron].getOutVal() *
                priorLayer[priorNeuron].m_outWeights[m_index].weight;
    }
    //This function shapes the output value of the neuron.
    //this has very limited scope and is actually quite simple
    //so it can be a private static function of class neuron.
    m_outVal = Neuron::activationFunction(sum);
}

Neuron::Neuron(unsigned numberOfOuts, unsigned localIndex) {
    for(unsigned connectLoop = 0; connectLoop < numberOfOuts; ++connectLoop){
        m_outWeights.push_back(Connection());
        m_outWeights.back().weight = randomWeight();
    }
    
    m_index = localIndex;
}


// ************** Class Net ********** //
//This lets us hold all our Neurons and
//do stuff with them.
class Net
{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals); //back propagation
    void getResults(vector<double> &resultVals) const;
    double getErrorAverage(void) const { return m_recentAverageError; }
private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error; // RMSE
    double m_recentAverageError;
    static double m_averageErrorSmoothingFactor;
};

//we can experiment with this, but it should be 1/10th the size
//of the sample size or so. For the sake of the pcc unix server
//we will limit ourselves to 2000 samples for now.
double Net::m_averageErrorSmoothingFactor = 200.0;


void Net::getResults(vector<double> &resultVals) const{
    resultVals.clear();
    
    for(unsigned nOutLoop = 0; nOutLoop < m_layers.back().size() - 1; ++nOutLoop){
        resultVals.push_back(m_layers.back()[nOutLoop].getOutVal());
    }
}

//This is how our net "learns" to handle data
//This is a "back Propagation" neural net.
void Net::backProp(const vector<double> &targetVals){
    //This will calculate the overall net errors
    //which will use a Root Mean Square Error to
    //calculate the difference between the
    //estimator and what is estimated. 
    //shortened as RMSE.
    Layer &outLayer = m_layers.back();
    m_error = 0.0;
    
    for(unsigned neuronPropLoop = 0; neuronPropLoop < outLayer.size() - 1;
             ++neuronPropLoop){
        double delta = targetVals[neuronPropLoop] - 
            outLayer[neuronPropLoop].getOutVal();
    m_error += delta * delta;

    }
    m_error /= outLayer.size() - 1; //this gets us the average error squared.
    m_error = sqrt(m_error); //this gets us RMSE.
    
    // This gives a recent average.
    m_recentAverageError =
        (m_recentAverageError * m_averageErrorSmoothingFactor + m_error)
        / (m_averageErrorSmoothingFactor + 1.0);
    //It will also calculate output/hidden layer
    //gradients.

    //Output
    for(unsigned nOutLoop = 0; nOutLoop < outLayer.size() - 1; ++nOutLoop) {
        outLayer[nOutLoop].outGradients(targetVals[nOutLoop]);
    }
    //hidden
    for (unsigned nHiddenLoop = m_layers.size() - 2; 
            nHiddenLoop > 0; --nHiddenLoop){
        Layer &hiddenLayer = m_layers[nHiddenLoop];
        Layer &nextLayer = m_layers[nHiddenLoop + 1];
        
        for(unsigned hLoop = 0; hLoop < hiddenLayer.size(); ++hLoop){
            hiddenLayer[hLoop].hiddenGradients(nextLayer);
        }
    
    }
    //As well as update connection weights.
    
    for(unsigned layerNumber = m_layers.size() - 1; layerNumber > 0; --layerNumber){
        Layer &layer = m_layers[layerNumber];
        Layer &priorLayer = m_layers[layerNumber - 1];
        
        //index neurons in layer.
        for (unsigned wNeuronLoop = 0; wNeuronLoop < layer.size() - 1; ++wNeuronLoop){
            layer[wNeuronLoop].updateInputWeights(priorLayer);
        }
    }
    
}

//feedForward is "feeding" information to the
//next neuron layer(s), as opposed to creating
//a directed cycle. This is one of the simplest
//types of artificial networks.
void Net::feedForward(const vector<double> &inputVals){
    assert(inputVals.size() == m_layers[0].size() - 1 );
    //this checks that our input values match up with
    //our input neurons. The -1 accounts for the bias
    //neuron.
    
    for(unsigned inputLoop = 0; inputLoop < inputVals.size(); ++inputLoop) {
        m_layers[0][inputLoop].setOutVal(inputVals[inputLoop]);
    }

    //This is where the forward propagation
    //itself takes place. We are looping through
    //each layer, and each neuron in each layer
    //telling each neuron to feed forward.
    for(unsigned layerNumber = 1; layerNumber < m_layers.size(); ++layerNumber) {
        Layer &priorLayer = m_layers[layerNumber - 1];
        
        for (unsigned neuronLoop = 0; neuronLoop < m_layers[layerNumber].size() - 1;
                ++neuronLoop) {
            m_layers[layerNumber][neuronLoop].feedForward(priorLayer);
            }
    }
};

//Our network of nodes/Neurons
//Also handles our layers.
Net::Net(const vector<unsigned> &topology)
{
    unsigned numberOfLayers = topology.size();
    for (unsigned layerNumber = 0; layerNumber < numberOfLayers; ++layerNumber) {
        m_layers.push_back(Layer()); //create new empty layer and append.
        unsigned numberOfOuts = layerNumber == topology.size() - 1 ? 0 : topology[layerNumber + 1];
        
        //Add new layer
        //Add a bias layer.
        for(unsigned neuronNumber = 0; neuronNumber <= topology[layerNumber]; 
            ++neuronNumber){
            m_layers.back().push_back(Neuron(numberOfOuts, neuronNumber));
            cout << "Layer 2 Neuron number: " << neuronNumber << " has been created" << endl;
            //feedback on how many hidden neurons are in the net
        }
     m_layers.back().back().setOutVal(1.0);
    }
}

void getVectorVal(string label, vector<double> &localVector)
{
    cout << label << " ";
    for (unsigned iterator = 0; iterator < localVector.size(); ++iterator) {
        cout << localVector[iterator] << " ";
    }

    cout << endl;
}

int main()
{
    //Grabs our training data file.
    Learn practiceData("./data/trainingSet1.txt");
    cout << "Beginning Neural Net training." << endl;
    cout << "Neuron creation announcements are for hidden layer neurons." << endl;
    cout << "Error rate should drop below 0.1 by training completion." << endl;
    cout << "Outputs will never exactly equal target outputs,"
       <<  " but will come very close."
       << endl;
    cout << "Accordingly values within 0.1 of targets are considered valid output" 
        << endl;
    // Ex. { 2, 6, 2 } in our case.
    vector<unsigned> topology;
    practiceData.getTopology(topology);

    Net localNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int learningLoop = 0;
    
    while (!practiceData.eofCheck()) {
        ++learningLoop;
        cout << endl << "Round " << learningLoop;

        // Fetch new input to feed forward
        if (practiceData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        getVectorVal(": Input(s):", inputVals);
        localNet.feedForward(inputVals);

        // Output results from net
        localNet.getResults(resultVals);
        getVectorVal("Output:", resultVals);

        // Teaching the net what we want the output to be.
        practiceData.getTargetOuts(targetVals);
        getVectorVal("Target:", targetVals);
        assert(targetVals.size() == topology.back());

        localNet.backProp(targetVals);

        // Report the error rate as training progresses.
        cout << "Net average error: "
            << localNet.getErrorAverage() << endl;
    }

    cout << endl << "Neural Network Training Completed." << endl;
}
