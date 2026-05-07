/**
 * Debug LCD Driver — Stub
 *
 * TODO: Implement after LCD model and interface are determined.
 *       Likely SPI-based (ST7735/ILI9341) or I2C OLED (SSD1306).
 */

#include "lcd_driver.h"

void LCD_Init(void)
{
    // TODO: Initialize LCD peripheral (SPI/I2C), send init sequence
}

void LCD_Clear(void)
{
    // TODO: Clear display buffer / send clear command
}

void LCD_DisplayJoints(const uint16_t angles[TOTAL_JOINTS])
{
    (void)angles;
    // TODO: Format and render 20 joint values on screen
    //
    // Suggested layout (for 128x160 or similar):
    //   Thumb:  DIP=xxxx PIP=xxxx MF=xxxx MB=xxxx
    //   Index:  DIP=xxxx PIP=xxxx MF=xxxx MB=xxxx
    //   Middle: DIP=xxxx PIP=xxxx MF=xxxx MB=xxxx
    //   Ring:   DIP=xxxx PIP=xxxx MF=xxxx MB=xxxx
    //   Pinky:  DIP=xxxx PIP=xxxx MF=xxxx MB=xxxx
}
