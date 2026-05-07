#ifndef __ADS1115_H
#define __ADS1115_H

#include "stm32f1xx_hal.h"
#include <stdint.h>

/* ---- ADS1115 Register Addresses ---- */
#define ADS1115_REG_CONVERSION  0x00
#define ADS1115_REG_CONFIG      0x01

/* ---- ADS1115 Config Register Bits ---- */
// Operational status / single-shot
#define ADS1115_OS_BEGIN        (1 << 15)

// Mux: input channel selection (bits 14:12)
#define ADS1115_MUX_AIN0_GND   (0x04 << 12)  // AIN0 vs GND
#define ADS1115_MUX_AIN1_GND   (0x05 << 12)
#define ADS1115_MUX_AIN2_GND   (0x06 << 12)
#define ADS1115_MUX_AIN3_GND   (0x07 << 12)

// PGA: gain (bits 11:9) — ±4.096V default
#define ADS1115_PGA_4_096V     (0x01 << 9)

// Mode: single-shot
#define ADS1115_MODE_SINGLE    (1 << 8)

// Data rate: 860 SPS (bits 7:5)
#define ADS1115_DR_860SPS      (0x07 << 5)

/* ---- I2C Addresses (7-bit, left-shifted by HAL) ---- */
// Bus 1: Thumb=0x48, Index=0x49, Middle=0x4A
// Bus 2: Ring=0x48, Pinky=0x49
#define ADS1115_ADDR_GND       (0x48 << 1)  // ADDR -> GND
#define ADS1115_ADDR_VDD       (0x49 << 1)  // ADDR -> VDD
#define ADS1115_ADDR_SDA       (0x4A << 1)  // ADDR -> SDA
#define ADS1115_ADDR_SCL       (0x4B << 1)  // ADDR -> SCL

/* ---- ADC Instance ---- */
typedef struct {
    I2C_HandleTypeDef *hi2c;
    uint8_t addr;  // 7-bit address, left-shifted
} ADS1115_Handle;

/* ---- API ---- */

// Initialize ADS1115 (verify communication)
HAL_StatusTypeDef ADS1115_Init(ADS1115_Handle *dev);

// Read single channel (0-3), blocking. Returns raw 16-bit signed value.
HAL_StatusTypeDef ADS1115_ReadChannel(ADS1115_Handle *dev, uint8_t channel, int16_t *value);

// Read all 4 channels into values[4], blocking.
HAL_StatusTypeDef ADS1115_ReadAllChannels(ADS1115_Handle *dev, int16_t values[4]);

#endif /* __ADS1115_H */
