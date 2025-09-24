/**
 * @brief Interface for linear regression algorithms.
 */

namespace ml 
{ 
namespace linreg
{
class Interface
{
public:
    /**
     * @brief Delete the constructor.
     */     
    virtual ~Interface() noexcept = default;

    /**
     * @brief Function to return the amount of epochs used.
         * 
     * @return Return the value of the variable epochsused.
     */
    virtual int getEpochsUsed() const noexcept = 0;

    /**
     * @brief Predict module.
     * 
     * @param[in] input The given data is what the module should base it´s predict on.
     * 
     * @return The predict value of given data.
     */
    virtual double predict(const double input) const = 0;

    /**
     * @brief Function to return the randomised startvalue of myBias.
     * 
     * @return Startvalue of myBias.
     */
    virtual double getBias() const noexcept = 0;

    /**
     * @brief Function to return the randomised startvalue of myWeight.
     * 
     * @return Startvalue of myWeight.
     */
    virtual double getWeight() const noexcept = 0;
};
} // Namespace lin_reg
} // namepsace mml