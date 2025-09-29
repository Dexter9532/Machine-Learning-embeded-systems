#include <stdlib.h> // For rand and srand.
#include <time.h>   // For time.

#include "driver/serial/interface.h"
#include "ml/linreg/linreg.h"
#include "driver/serial/interface.h"
#include "container/vector.h"

namespace ml
{
namespace linreg
{
namespace
{
/**
 * @brief Initialize the random generator (once only).
 */
void initRandom() noexcept
{
    // Create a static local variable, which indicates whether the generator has been initialized.
    // This line (with the initialization) is only run once.
    static auto initialized{false};

    // Terminate the function if the generator already has been initialized.
    if (initialized) { return; }

    // Initialize the generator, use the current time as seed (start of the random sequence).
    // Get the current time via time(nullptr).
    srand(time(nullptr));

    // Mark the random generator as initialized.
    initialized = true;
}

/**
 * @brief Get a random starting value for the linear regression parameters.
 * 
 * @return Random floating-point number between 0.0 - 1.0.
 */
double randomStartVal() noexcept
{
    // Divide rand() by RAND_MAX, cast RAND_MAX to double to ensure floating-point division.
    return rand() / static_cast<double>(RAND_MAX);
}

/**
 * @brief Shuffle the content of the given vector.
 * 
 * @param[in, out] data Reference to the vector to shuffle.
 */
void shuffle(container::Vector<size_t>& data) noexcept
{
    // Shuffle the vector by swapping each element with a random element.
    for (size_t i{}; i < data.size(); ++i)
    {
        // Get a random index r (between 0-4 if we have five training sets).
        const auto r{rand() % data.size()};

        // Swap the elements at index i and r => make a copy of data[i].
        const auto temp{data[i]};

        // Copy data[r] to data[i] => now we have two instances of data[r] in the vector.
        data[i] = data[r];

        // Finally put the copy of the "old" data[i] to data[r] => we have swapped the elements.
        data[r] = temp;
    }
}

/**
 * @brief calculate absolute value of a double
 * 
 * @param [in] x input value
 * 
 * @return non-negative absolute value of x
 */
static inline double dabs(double x) 
{
    return (x < 0.0) ? -x : x;
}

//--------------------------------------------------------------------------------//
constexpr size_t min(const size_t x, const size_t y) noexcept
{
    // Return x if x <= y, else y.
    return x <= y ? x : y;
}
} // namespace

//--------------------------------------------------------------------------------//
LinReg::LinReg(const container::Vector<double>& trainInput,
               const container::Vector<double>& trainOutput,
               driver::SerialInterface& serial) noexcept
                :   myTrainInput{trainInput},
                    myTrainOutput{trainOutput},  
                    myTrainSetCount{min(trainInput.size(), trainOutput.size())},
                    myPredVector(myTrainSetCount),
                    mySerial{serial}
{
    mySerial.setEnabled(true);
    // Random generator and uniform.
    initRandom();

    // Assign random values.
    myBias = randomStartVal();
    myWeight = randomStartVal();

    myIndex.resize(myTrainSetCount); // Tell vector how many elements it will contain to not allocate vector. 

    // Loop to add the indexes in the trainingdata to the vector myIndex.
    for (size_t i{0}; i < myTrainSetCount; i++)
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

    while (1)
    {
        shuffle(myIndex);

        for (size_t k{}; k < myTrainSetCount; k++)
        {
            const size_t i = myIndex[k];
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
        if (isPredictDone()) { break; }
    }
    return true;    
}
//--------------------------------------------------------------------------------//
bool LinReg::isPredictDone() const noexcept
{
    constexpr double tol = 1e-4;
    for (size_t i{}; i < myTrainSetCount; ++i)
    {
        if (dabs(myPredVector[i] - myTrainOutput[i]) > tol)
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

} //namespace linreg
} //namespace ml

