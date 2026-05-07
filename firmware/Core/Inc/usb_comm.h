#ifndef __USB_COMM_H
#define __USB_COMM_H

#include "main.h"
#include <stdint.h>

/* ---- API ---- */

// Initialize USB CDC (called after HAL USB init)
void USB_Comm_Init(void);

// Send a complete joint data frame over USB CDC
// Returns 0 on success, -1 on failure
int USB_Comm_SendFrame(const JointDataFrame *frame);

// Build frame from raw angle array, compute checksum, and send
int USB_Comm_PackAndSend(const uint16_t angles[TOTAL_JOINTS]);

#endif /* __USB_COMM_H */
