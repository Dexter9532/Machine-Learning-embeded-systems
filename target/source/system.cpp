/**
 * @brief Generic system implementation details for an MCU with configurable hardware devices.
 */
#include <stdint.h>

#include "driver/adc/interface.h"
#include "driver/eeprom/interface.h"
#include "driver/gpio/interface.h"
#include "driver/serial/interface.h"
#include "driver/timer/interface.h"
#include "driver/watchdog/interface.h"
#include "target/system.h"
#include "ml/linreg/interface.h"

namespace target
{
namespace
{
constexpr int round(const double number)
{
    // Case 1: number = 2.7 => we cast 2.7 + 0.5 to int => 3.2 is converted to 3.
    // Case 2: number = 2.3 => we cast 2.3 + 0.5 to int => 2.8 is converted to 2.
    // Case 3: number = -4.7 => we cast -4.7 - 0.5 to int => -5.2 is converted to -5.
    // Case 4: number = -4.2 => we cast -4.2 - 0.5 to int => -4.7 is converted to -4.
    return 0.0 <= number ? static_cast<int>(number + 0.5) : static_cast<int>(number - 0.5);
}
} // namespace
/**
 * @brief Structure of LED state parameters.
 */
namespace LedState
{
    /** LED state address in EEPROM. */
    static constexpr uint8_t address{0U};

    /** Enabled state value in EEPROM. */
    static constexpr uint8_t enabled{1U};
};

// -----------------------------------------------------------------------------
System::System(driver::GpioInterface& led, driver::GpioInterface& button,
               driver::TimerInterface& debounceTimer, driver::TimerInterface& predictTimer,
               driver::SerialInterface& serial, driver::WatchdogInterface& watchdog,
               driver::EepromInterface& eeprom, driver::AdcInterface& adc,
               ml::linreg::Interface& linReg, const uint8_t sensorPin) noexcept
    : myLed{led}
    , myButton{button}
    , myDebounceTimer{debounceTimer}
    , myPredictTimer{predictTimer}
    , mySerial{serial}
    , myWatchdog{watchdog}
    , myEeprom{eeprom}
    , myAdc{adc}
    , myLinReg{linReg}
    , mySensorPin{sensorPin}
{
    myButton.enableInterrupt(true);
    mySerial.setEnabled(true);
    myWatchdog.setEnabled(true);
    myPredictTimer.start();
    myAdc.setEnabled(true);
}

// -----------------------------------------------------------------------------
System::~System() noexcept
{
    myLed.write(false);
    myButton.enableInterrupt(false);
    myDebounceTimer.stop();
    myPredictTimer.stop();
    myWatchdog.setEnabled(false);
}

// -----------------------------------------------------------------------------
void System::enableSerialTransmission(const bool enable) noexcept
{
    mySerial.setEnabled(enable);
}

// -----------------------------------------------------------------------------
void System::handleButtonInterrupt() noexcept
{
    myButton.enableInterruptOnPort(false);
    myDebounceTimer.start();
    if (myButton.read()) { handleButtonPressed(); }
}

// -----------------------------------------------------------------------------
void System::handleDebounceTimerInterrupt() noexcept
{
    myDebounceTimer.stop();
    myButton.enableInterruptOnPort(true);
}

// -----------------------------------------------------------------------------
void System::handlepredictTimerInterrupt() noexcept 
{
    // Prediktera temperaturen och skriv ut här.
    // Timern startar om sig själv, så tänk inte på det. 
}

// -----------------------------------------------------------------------------
void System::run() noexcept
{
    mySerial.printf("Running the system!\n");
    
    while (1)
    {
        myWatchdog.reset();
    }
}

// -----------------------------------------------------------------------------
void System::handleButtonPressed() noexcept
{
    mySerial.printf("Button pressed!\n");

    // Läs av ADC, prediktera temperaturen och skriv ut den.
    // Nollställ också 60-sekunderstimern.

    const auto inputVoltage{myAdc.inputVoltage(mySensorPin)};
    const auto prediction{myLinReg.predict(inputVoltage)};
    
    mySerial.printf("The temperature is: %d \n", round(prediction));
    myPredictTimer.restart();
}
} // namespace target