import UIKit
import WebKit

/// MusicKit JS calls window.open() for Apple Music auth, which normally opens
/// in external Safari. Safari is a separate process and cannot postMessage back
/// to the Capacitor WKWebView, so auth never completes. This handler intercepts
/// the popup and shows it inside the app so postMessage works correctly.
class MusicKitPopupHandler: NSObject, WKUIDelegate {
    weak var originalDelegate: WKUIDelegate?
    private var popupWebView: WKWebView?
    private var popupViewController: UIViewController?

    func webView(
        _ webView: WKWebView,
        createWebViewWith configuration: WKWebViewConfiguration,
        for navigationAction: WKNavigationAction,
        windowFeatures: WKWindowFeatures
    ) -> WKWebView? {
        let popup = WKWebView(frame: .zero, configuration: configuration)
        popup.uiDelegate = self

        let vc = UIViewController()
        vc.view.backgroundColor = UIColor(red: 0.04, green: 0.04, blue: 0.04, alpha: 1)

        popup.translatesAutoresizingMaskIntoConstraints = false
        vc.view.addSubview(popup)
        NSLayoutConstraint.activate([
            popup.topAnchor.constraint(equalTo: vc.view.safeAreaLayoutGuide.topAnchor),
            popup.bottomAnchor.constraint(equalTo: vc.view.bottomAnchor),
            popup.leadingAnchor.constraint(equalTo: vc.view.leadingAnchor),
            popup.trailingAnchor.constraint(equalTo: vc.view.trailingAnchor),
        ])

        let rootVC = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.keyWindow?.rootViewController }
            .first
        rootVC?.present(vc, animated: true)

        popupWebView = popup
        popupViewController = vc
        return popup
    }

    func webViewDidClose(_ webView: WKWebView) {
        popupViewController?.dismiss(animated: true)
        popupWebView = nil
        popupViewController = nil
    }

    // Forward JS dialog methods to original Capacitor delegate
    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        if let delegate = originalDelegate {
            delegate.webView?(webView, runJavaScriptAlertPanelWithMessage: message, initiatedByFrame: frame, completionHandler: completionHandler)
        } else {
            completionHandler()
        }
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        if let delegate = originalDelegate {
            delegate.webView?(webView, runJavaScriptConfirmPanelWithMessage: message, initiatedByFrame: frame, completionHandler: completionHandler)
        } else {
            completionHandler(false)
        }
    }

    func webView(_ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String, defaultText: String?, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (String?) -> Void) {
        if let delegate = originalDelegate {
            delegate.webView?(webView, runJavaScriptTextInputPanelWithPrompt: prompt, defaultText: defaultText, initiatedByFrame: frame, completionHandler: completionHandler)
        } else {
            completionHandler(nil)
        }
    }
}
