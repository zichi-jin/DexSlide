#ifndef __LCD_DRIVER_H
#define __LCD_DRIVER_H

#include "main.h"
#include <stdint.h>

/*
 * Debug LCD driver — interface TBD.
 *
 * TODO: Determine LCD model and connection interface:
 *   - SPI (ST7735, ILI9341, etc.)
 *   - I2C (SSD1306 OLED, etc.)
 *   - Parallel 8080
 * Pin assignments depend on the chosen LCD.
 */

/* ---- API ---- */

// Initialize LCD hardware
void LCD_Init(void);

// Clear screen
void LCD_Clear(void);

// Display all 20 joint angles on screen
// angles: array of TOTAL_JOINTS raw values
void LCD_DisplayJoints(const uint16_t angles[TOTAL_JOINTS]);

#endif /* __LCD_DRIVER_H */
