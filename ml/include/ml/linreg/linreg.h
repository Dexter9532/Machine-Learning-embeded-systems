/**
 * @brief File to define the subclass named LinReg
 */
#pragma once

#include "ml/linreg/interface.h"
#include "container/vector.h"

namespace driver
{
// Forward declaration of serial implementation.
class SerialInterface;
}

namespace ml
{
namespace linreg
{
/**
 * @brief Class LinReg that inherits from the Class Interface
 * 
 * @note The class is final.
 */
class LinReg final: public Interface
{
public:
    /**
     * @brief Constructor LinReg.
     * 
     * @param [in] trainInput Reference to a readble vector, (data that is going to be traded).
     * @param [in] tainOutput Reference to a readble vector, (data that is going to be traded).
     */
    explicit LinReg(const container::Vector<double>& trainInput,
                    const container::Vector<double>& trainOutput,
                    driver::SerialInterface& serial) noexcept;

    /**
     * @brief Delete the constructor as default.
     */
    ~LinReg() noexcept override = default;

    /**
     * @brief Predict module
     * 
     * @param[in] input The given data is what the module should base it´s predict on.
     * 
     * @return The predict value of given data.
     */
    double predict(const double input) const noexcept override;
    
    /**
     * @brief Check if predict is right.
     * 
     * @return Return true if predict is right, return false outerwise.
     */
    bool isPredictDone() const noexcept;

    /**
     * @brief Method to train the module without setting epochcount.
     * 
     * @param [in] learingRate Learingrate speed, default is 0.01 or 1%.
     */
    bool trainWithNoEpoch(double learningRate = 0.01) noexcept;
    
    /**
     * @brief Function to return the amount of epochs used.
     * 
     * @return Return the value of the variable epochsused,
     * specified epochs are used then return the value of the given amount.
     */
    int getEpochsUsed() const noexcept override;
    
    /**
     * @brief Function to return the randomised startvalue of myBias.
     * 
     * @return Startvalue of myBias.
     */
    double getBias() const noexcept override;

    /**
     * @brief Function to return the randomised startvalue of myWeight.
     * 
     * @return Startvalue of myWeight.
     */
    double getWeight() const noexcept override;

    LinReg() = delete;                            // Delete a car without info.
    LinReg(const LinReg&) = delete;               // Delete copy constructor.
    LinReg& operator=(const LinReg&) = delete;    // Delete copy assignment.
    LinReg(LinReg&&) = delete;                    // Delete move constructor.
    LinReg& operator=(LinReg&&) = delete;         // Delete move assignment.

private:

    const container::Vector<double>& myTrainInput;            // Reference to the training data (input data).
    const container::Vector<double>& myTrainOutput;           // Reference to the training data (output data).
    size_t myTrainSetCount;                        // Indicates the total of full trainingset that are avalible.
    double myBias;                                      // Bias value for the module, (m) in the ecvation kx + m = y.
    double myWeight;                                    // Weight value for the module, (k) in the ecvation kc + m = y.
    container::Vector<double> myLastPredict;                  // Reference to the last data the modlue has between epochs.
    int myEpochsUsed{0};                                // To save the amount of epochs that are used for the specific traingmodule.
    container::Vector<double> myPredVector;                   // Last predict.
    size_t myEpochCount{0};                        // The specified amount of epochs that the module should use.   
    container::Vector<size_t> myIndex;                   // Vector that holds the amount of indexes in traingvector to shuffle.
    driver::SerialInterface& mySerial;

};
} // namespace linreg
} // namespace ml