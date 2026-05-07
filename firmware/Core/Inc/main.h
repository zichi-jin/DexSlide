#ifndef __MAIN_H
#define __MAIN_H

#include "stm32f1xx_hal.h"

/* ---- Pin Definitions ---- */
// I2C1: Thumb, Index, Middle ADCs
#define I2C1_SCL_PIN    GPIO_PIN_6
#define I2C1_SDA_PIN    GPIO_PIN_7
#define I2C1_GPIO_PORT  GPIOB

// I2C2: Ring, Pinky ADCs
#define I2C2_SCL_PIN    GPIO_PIN_10
#define I2C2_SDA_PIN    GPIO_PIN_11
#define I2C2_GPIO_PORT  GPIOB

/* ---- Constants ---- */
#define NUM_FINGERS     5
#define JOINTS_PER_FINGER 4
#define TOTAL_JOINTS    (NUM_FINGERS * JOINTS_PER_FINGER)  // 20

/* ---- Data Frame ---- */
// Packed frame sent over USB CDC
// [HEADER(2)] [20x uint16_t angles(40)] [CHECKSUM(1)] [TAIL(1)]
#define FRAME_HEADER    0xAA55
#define FRAME_SIZE      (2 + TOTAL_JOINTS * 2 + 1 + 1)  // 44 bytes

typedef struct {
    uint16_t header;
    uint16_t angles[TOTAL_JOINTS];  // Raw ADC values (0-65535 for ADS1115)
    uint8_t  checksum;
    uint8_t  tail;
} __attribute__((packed)) JointDataFrame;

/* ---- Finger/Joint Index Convention ---- */
// angles[0..3]   = Thumb:  DIP, PIP, MCP_front, MCP_back
// angles[4..7]   = Index:  DIP, PIP, MCP_front, MCP_back
// angles[8..11]  = Middle: DIP, PIP, MCP_front, MCP_back
// angles[12..15] = Ring:   DIP, PIP, MCP_front, MCP_back
// angles[16..19] = Pinky:  DIP, PIP, MCP_front, MCP_back

/* ---- Function Prototypes ---- */
void SystemClock_Config(void);
void Error_Handler(void);

#endif /* __MAIN_H */
