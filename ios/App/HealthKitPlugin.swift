import Foundation
import Capacitor
import HealthKit

@objc(HealthKitPlugin)
public class HealthKitPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "HealthKitPlugin"
    public let jsName = "HealthKit"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "requestPermission", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "startHeartRateMonitoring", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "stopHeartRateMonitoring", returnType: CAPPluginReturnPromise),
    ]

    private let healthStore = HKHealthStore()
    private var pollTimer: Timer?

    @objc func requestPermission(_ call: CAPPluginCall) {
        guard HKHealthStore.isHealthDataAvailable() else {
            call.reject("HealthKit not available on this device")
            return
        }
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            call.reject("Heart rate type unavailable")
            return
        }
        healthStore.requestAuthorization(toShare: nil, read: [heartRateType]) { granted, error in
            if let error = error {
                call.reject(error.localizedDescription)
                return
            }
            call.resolve(["granted": granted])
        }
    }

    @objc func startHeartRateMonitoring(_ call: CAPPluginCall) {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            call.reject("Heart rate type unavailable")
            return
        }

        // Fetch immediately on start
        fetchLatestHeartRate(heartRateType: heartRateType)

        // Poll every 5 seconds — grabs whatever the watch last recorded
        DispatchQueue.main.async {
            self.pollTimer?.invalidate()
            self.pollTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
                self?.fetchLatestHeartRate(heartRateType: heartRateType)
            }
        }

        call.resolve()
    }

    @objc func stopHeartRateMonitoring(_ call: CAPPluginCall) {
        DispatchQueue.main.async {
            self.pollTimer?.invalidate()
            self.pollTimer = nil
        }
        call.resolve()
    }

    private func fetchLatestHeartRate(heartRateType: HKQuantityType) {
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        let query = HKSampleQuery(
            sampleType: heartRateType,
            predicate: nil,
            limit: 1,
            sortDescriptors: [sortDescriptor]
        ) { [weak self] _, samples, _ in
            guard let sample = samples?.first as? HKQuantitySample else { return }
            let bpm = sample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            self?.notifyListeners("heartRateUpdate", data: ["bpm": Int(bpm)])
        }
        healthStore.execute(query)
    }
}
