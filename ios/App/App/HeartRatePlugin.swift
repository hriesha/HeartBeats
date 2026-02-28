import Foundation
import Capacitor
import HealthKit

@objc(HeartRatePlugin)
public class HeartRatePlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "HeartRatePlugin"
    public let jsName = "HeartRate"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "isAvailable", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "requestAuthorization", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getLatestHeartRate", returnType: CAPPluginReturnPromise),
    ]

    private let healthStore = HKHealthStore()

    @objc func isAvailable(_ call: CAPPluginCall) {
        call.resolve(["available": HKHealthStore.isHealthDataAvailable()])
    }

    @objc func requestAuthorization(_ call: CAPPluginCall) {
        guard HKHealthStore.isHealthDataAvailable() else {
            call.reject("HealthKit not available on this device")
            return
        }

        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        healthStore.requestAuthorization(toShare: nil, read: [heartRateType]) { success, error in
            if let error = error {
                call.reject("Authorization failed: \(error.localizedDescription)")
            } else {
                call.resolve(["authorized": success])
            }
        }
    }

    @objc func getLatestHeartRate(_ call: CAPPluginCall) {
        guard HKHealthStore.isHealthDataAvailable() else {
            call.reject("HealthKit not available")
            return
        }

        let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sortDescriptor]
        ) { _, results, error in
            if let error = error {
                call.reject("Query failed: \(error.localizedDescription)")
                return
            }

            guard let sample = results?.first as? HKQuantitySample else {
                call.resolve(["bpm": nil, "timestamp": nil])
                return
            }

            let bpm = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            let timestamp = sample.startDate.timeIntervalSince1970 * 1000

            call.resolve([
                "bpm": Int(bpm),
                "timestamp": timestamp
            ])
        }

        healthStore.execute(query)
    }
}
