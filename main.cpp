/**
 * @brief Demonstration of GPIO device drivers in C++:
 * 
 *        The following devices are used:
 *            - A button connected to pin 13 on the device toggles a timer.
 *            - The aforementioned timer toggles an LED every 100 ms when enabled.
 *            - Another timer reduces the effect of contact bounces after pushing the button.
 *            - A watchdog timer is used to restart the program if it gets stuck somewhere.
 *            - An EEPROM stream is used to store the LED state. On startup, this value is read;
 *              if the last stored state before power down was "on," the LED will automatically blink.
 */
#include "driver/atmega328p/adc.h"
#include "driver/atmega328p/eeprom.h"
#include "driver/atmega328p/gpio.h"
#include "driver/atmega328p/serial.h"
#include "driver/atmega328p/timer.h"
#include "driver/atmega328p/watchdog.h"
#include "target/system.h"
#include "ml/linreg/linreg.h"

using namespace driver::atmega328p;

namespace
{
/** Pointer to the system implementation. */
target::System* mySys{nullptr};
    
// Obtain a reference to the singleton serial device instance.
auto& serial{Serial::getInstance()};

/**
 * @brief Callback for the button.
 */
void buttonCallback() noexcept { mySys->handleButtonInterrupt(); }

/**
 * @brief Callback for the debounce timer.
 * 
 *        This callback is invoked whenever the debounce timer elapses.
 */
void debounceTimerCallback() noexcept { mySys->handleDebounceTimerInterrupt(); }

/**
 * @brief Callback for the toggle timer.
 * 
 *        This callback is invoked whenever the toggle timer elapses.
 */
void toggleTimerCallback() noexcept { mySys->handleToggleTimerInterrupt(); }

constexpr int round(const double number)
{
    // Case 1: number = 2.7 => we cast 2.7 + 0.5 to int => 3.2 is converted to 3.
    // Case 2: number = 2.3 => we cast 2.3 + 0.5 to int => 2.8 is converted to 2.
    // Case 3: number = -4.7 => we cast -4.7 - 0.5 to int => -5.2 is converted to -5.
    // Case 4: number = -4.2 => we cast -4.2 - 0.5 to int => -4.7 is converted to -4.
    return 0.0 <= number ? static_cast<int>(number + 0.5) : static_cast<int>(number - 0.5);
}

/**
 * @brief Predict with the given linear regression model.
 * 
 * @param[in] linreg Linear regression model to predict with.
 * @param[in] inputData Input data to predict with.
 */
void printPredictions(const ml::linreg::Interface& linReg, const container::Vector<double>& inputData) noexcept
{
    // Terminate the function if no input data is provided.
    if (inputData.empty())
    {
        serial.printf("No input data!\n");
        return;
    }
    serial.printf("--------------------------------------------------------------------------------\n");
    // Perform prediction with each input value, print the result in the terminal.
    for (const auto& input : inputData)
    {
        const auto prediction{linReg.predict(input)};
        const auto mV{input * 1000.0};
        serial.printf("Input: %d, predicted output: %d mV\n", round(mV), round(prediction));
    }
    serial.printf("Epochs used: %d\n", linReg.getEpochsUsed());
    serial.printf("--------------------------------------------------------------------------------\n\n");
}
} // namespace

/**
 * @brief Initialize and run the system on the target MCU.
 * 
 * @return 0 on termination of the program (should never occur).
 */

int main()
{
    serial.setEnabled(true);
    
    serial.printf("Hello there!");
        
    // Learingrate for the training.
    constexpr double learningRate{0.225};

    // The data we want to train our model with.
    const container::Vector<double> trainInput{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    const container::Vector<double> trainOutput{-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0};

    // The constructor.
    ml::linreg::LinReg linReg{trainInput, trainOutput};
    if (!linReg.trainWithNoEpoch(serial, learningRate)) {
        serial.printf("Training failed!\n");
        return -1;
    }
    serial.printf("Training finished!\n");
    printPredictions(linReg, trainInput);
    return 0;
    // Gällande ADC:
    // read returnerar ett värde mellan 0 - 1023.
    // dutyCycle returnerar ett värde mellan 0.0 - 1.0 (den tar ADC-värdet / 1023.0).
    // inputVoltage tar duty_cycle * spänningen, så den returnerar motsvarande värde mellan 0 - 5 V.

    serial.printf("Hello, Jobo!\n");
    // Skapa och träna LinReg-modellen här.

    // Initialize the GPIO devices.
    Gpio led{9U, Gpio::Direction::Output};
    Gpio button{8U, Gpio::Direction::InputPullup, buttonCallback};

    // Initialize the timers.
    Timer debounceTimer{300U, debounceTimerCallback};
    Timer toggleTimer{100U, toggleTimerCallback};

    // Obtain a reference to the singleton watchdog timer instance.
    auto& watchdog{Watchdog::getInstance()};

    // Obtain a reference to the singleton EEPROM instance.
    auto& eeprom{Eeprom::getInstance()};

    // Obtain a reference to the singleton ADC instance.
    auto& adc{Adc::getInstance()};

    // Initialize the system with the given hardware.
    // Skicka med din LinReg-modell till system-klassen, där i körs prediktion etc.
    target::System system{led, button, debounceTimer, toggleTimer, serial, watchdog, eeprom, adc, linReg};
    mySys = &system;

    // Run the system perpetually on the target MCU.
    mySys->run();

    // This point should never be reached; the system is intended to run indefinitely on the target MCU.
    return 0;
}
