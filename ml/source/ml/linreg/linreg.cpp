#include "ml/lin_reg/lin_reg.h"
#include <vector>
#include <cmath>
#include <algorithm> // for std::shuffle
#include <random>
#include <iostream>

namespace ml::lin_reg
{
//--------------------------------------------------------------------------------//
LinReg::LinReg(const std::vector<double>& trainInput,
               const std::vector<double>& trainOutput) noexcept
                :   myTrainInput{trainInput},
                    myTrainOutput{trainOutput},  
                    myTrainSetCount(static_cast<unsigned>(
                    std::min(trainInput.size(), trainOutput.size()))),
                    myPredVector(myTrainSetCount)
{
    // Random generator and uniform.
    static std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<double> dist (0.0, 1.0);

    // Assign random values.
    myBias = dist(gen);
    myWeight = dist(gen);
    
    // Print the startvalues.
    std::cout << "Bias startvalue: " << myBias << "\n";
    std::cout << "Weight startvalue: " << myWeight << "\n";

    myIndex.resize(myTrainSetCount); // Tell vector how many elements it will contain to not allocate vector. 

    // Loop to add the indexes in the trainingdata to the vector myIndex.
    for (std::size_t i{0}; i < myTrainSetCount; i++)
    {
        myIndex[i] = i;
    }
}   
//--------------------------------------------------------------------------------//
double LinReg::predict(const double input) const noexcept
{
    return (myWeight * input + myBias);
}
//--------------------------------------------------------------------------------//
bool LinReg::trainWithNoEpoch(double learningRate) noexcept
{
    if ((0.0 >= learningRate)) { return false;}

    while (!isPredictDone())
    {
        shuffleIndex();
        for (std::size_t k{}; k < myTrainSetCount; k++)
        {
            const std::size_t i = myIndex[k];
            // ypred = kx + m.
            const auto yPred = predict(myTrainInput[i]);

            // e = yref - ypred.
            const auto e = (myTrainOutput[i] - yPred);

            // m = m + e * LR.
            myBias = myBias + (e * learningRate);

            // k = k + e * LR * x.
            myWeight = myWeight + (e * learningRate * myTrainInput[i]); 
            
            myPredVector[i] = predict(myTrainInput[i]);
        }
        // Save epochs used.
        myEpochsUsed++;
    }
    return true;    
}
//--------------------------------------------------------------------------------//
bool LinReg::train(const std::size_t epochCount, double learningRate) noexcept
{
    if ((0U == epochCount) || (0.0 >= learningRate)) { return false;}
    
    myEpochCount = epochCount;   

    for (std::size_t epoch{}; epoch < epochCount; epoch++)
    {
        shuffleIndex();
        for (std::size_t k{}; k < myTrainSetCount; k++)
        {
            // Use random index.
            const std::size_t i = myIndex[k];

            // ypred = kx + m.
            const auto yPred = predict(myTrainInput[i]);

            // e = yref - ypred.
            const auto e = (myTrainOutput[i] - yPred);

            // m = m + e * LR.
            myBias = myBias + (e * learningRate);

            // k = k + e * LR * x.
            myWeight = myWeight + (e * learningRate * myTrainInput[i]); 
        }
    }
    return true;
}
//--------------------------------------------------------------------------------//
bool LinReg::isPredictDone() const noexcept
{
    constexpr double tol = 1e-6;
    for (std::size_t i{}; i < myTrainSetCount; ++i)
    {
        if (std::abs(myPredVector[i] - myTrainOutput[i]) > tol)
        {
            return false;
        }
    }
    return true;
}
//--------------------------------------------------------------------------------//
int LinReg::getEpochsUsed() const noexcept 
{
    if (myEpochsUsed == 0)
    { 
        return myEpochCount;
    } 
    return myEpochsUsed;
}
//--------------------------------------------------------------------------------//
double LinReg::getBias() const noexcept {return myBias; }
//--------------------------------------------------------------------------------//
double LinReg::getWeight() const noexcept {return myWeight; }
//--------------------------------------------------------------------------------//
void LinReg::shuffleIndex() noexcept
{
    static std::mt19937 gen{std::random_device{}()};
    std::shuffle(myIndex.begin(), myIndex.end(), gen);
}
} //namespace ml::lin_reg

