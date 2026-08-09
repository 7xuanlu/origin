// SPDX-License-Identifier: Apache-2.0

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#include <stdint.h>

int32_t wenlan_macos_thermal_state(void) {
    return (int32_t)NSProcessInfo.processInfo.thermalState;
}

double wenlan_macos_seconds_since_last_input(void) {
    return CGEventSourceSecondsSinceLastEventType(
        kCGEventSourceStateCombinedSessionState,
        kCGAnyInputEventType
    );
}
