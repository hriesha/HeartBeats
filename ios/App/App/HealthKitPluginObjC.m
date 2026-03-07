#import <Foundation/Foundation.h>
#import <Capacitor/Capacitor.h>

CAP_PLUGIN(HealthKitPlugin, "HealthKit",
    CAP_PLUGIN_METHOD(requestPermission, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(startHeartRateMonitoring, CAPPluginReturnPromise);
    CAP_PLUGIN_METHOD(stopHeartRateMonitoring, CAPPluginReturnPromise);
)
