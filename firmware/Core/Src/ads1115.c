/**
 * ADS1115 Driver — I2C 16-bit ADC
 *
 * Single-shot mode, 860 SPS, ±4.096V range.
 * Reads 4 single-ended channels (AINx vs GND).
 */

#include "ads1115.h"

/* ---- MUX values for each channel ---- */
static const uint16_t channel_mux[4] = {
    ADS1115_MUX_AIN0_GND,
    ADS1115_MUX_AIN1_GND,
    ADS1115_MUX_AIN2_GND,
    ADS1115_MUX_AIN3_GND,
};

/* ---- Default config: single-shot, 860SPS, ±4.096V ---- */
static const uint16_t base_config =
    ADS1115_OS_BEGIN |
    ADS1115_PGA_4_096V |
    ADS1115_MODE_SINGLE |
    ADS1115_DR_860SPS |
    0x0003;  // COMP_QUE: disable comparator

#define I2C_TIMEOUT_MS  50

HAL_StatusTypeDef ADS1115_Init(ADS1115_Handle *dev)
{
    // Verify device responds on I2C
    return HAL_I2C_IsDeviceReady(dev->hi2c, dev->addr, 3, I2C_TIMEOUT_MS);
}

HAL_StatusTypeDef ADS1115_ReadChannel(ADS1115_Handle *dev, uint8_t channel, int16_t *value)
{
    if (channel > 3) return HAL_ERROR;

    HAL_StatusTypeDef status;
    uint16_t config = base_config | channel_mux[channel];

    // Write config register to start conversion
    uint8_t tx[3];
    tx[0] = ADS1115_REG_CONFIG;
    tx[1] = (config >> 8) & 0xFF;
    tx[2] = config & 0xFF;
    status = HAL_I2C_Master_Transmit(dev->hi2c, dev->addr, tx, 3, I2C_TIMEOUT_MS);
    if (status != HAL_OK) return status;

    // Wait for conversion (~1.2ms at 860 SPS)
    HAL_Delay(2);

    // TODO: For higher throughput, poll the OS bit in config register
    // instead of using HAL_Delay. Or switch to continuous mode.

    // Read conversion result
    uint8_t reg = ADS1115_REG_CONVERSION;
    status = HAL_I2C_Master_Transmit(dev->hi2c, dev->addr, &reg, 1, I2C_TIMEOUT_MS);
    if (status != HAL_OK) return status;

    uint8_t rx[2];
    status = HAL_I2C_Master_Receive(dev->hi2c, dev->addr, rx, 2, I2C_TIMEOUT_MS);
    if (status != HAL_OK) return status;

    *value = (int16_t)((rx[0] << 8) | rx[1]);
    return HAL_OK;
}

HAL_StatusTypeDef ADS1115_ReadAllChannels(ADS1115_Handle *dev, int16_t values[4])
{
    for (uint8_t ch = 0; ch < 4; ch++) {
        HAL_StatusTypeDef status = ADS1115_ReadChannel(dev, ch, &values[ch]);
        if (status != HAL_OK) return status;
    }
    return HAL_OK;
}
