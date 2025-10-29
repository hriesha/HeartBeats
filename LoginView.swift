import SwiftUI

// Color extension for hex color support
extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a, r, g, b: UInt64
        switch hex.count {
        case 3: // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6: // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8: // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue:  Double(b) / 255,
            opacity: Double(a) / 255
        )
    }
}

struct LoginView: View {
    @State private var email = ""
    @State private var password = ""
    @State private var isPasswordVisible = false
    @State private var isLoading = false
    @State private var showAlert = false
    @State private var alertMessage = ""
    
    // Color palette
    private let backgroundColor = Color(hex: "03071E") // Darkest - background
    private let cardBackground = Color(hex: "370617")
    private let accentColor = Color(hex: "DC2F02")
    private let textColor = Color(hex: "FFBA08")
    private let secondaryTextColor = Color(hex: "FAA307")
    private let gradientStart = Color(hex: "6A040F")
    private let gradientEnd = Color(hex: "9D0208")
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background with gradient overlay
                backgroundColor
                    .ignoresSafeArea()
                
                // Gradient overlay for depth
                LinearGradient(
                    gradient: Gradient(colors: [gradientStart.opacity(0.3), gradientEnd.opacity(0.1)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 0) {
                        // Header section
                        VStack(spacing: 20) {
                            // Logo/Icon
                            ZStack {
                                Circle()
                                    .fill(
                                        LinearGradient(
                                            gradient: Gradient(colors: [accentColor, Color(hex: "E85D04")]),
                                            startPoint: .topLeading,
                                            endPoint: .bottomTrailing
                                        )
                                    )
                                    .frame(width: 120, height: 120)
                                    .shadow(color: accentColor.opacity(0.3), radius: 20, x: 0, y: 10)
                                
                                Image(systemName: "heart.fill")
                                    .font(.system(size: 50, weight: .bold))
                                    .foregroundColor(.white)
                                    .shadow(color: .black.opacity(0.3), radius: 5, x: 0, y: 2)
                            }
                            .padding(.top, 60)
                            
                            // App title
                            VStack(spacing: 8) {
                                Text("HeartBeats")
                                    .font(.system(size: 36, weight: .bold, design: .rounded))
                                    .foregroundColor(textColor)
                                    .shadow(color: .black.opacity(0.3), radius: 5, x: 0, y: 2)
                                
                                Text("Sync your music with your rhythm")
                                    .font(.system(size: 16, weight: .medium))
                                    .foregroundColor(secondaryTextColor)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 40)
                            }
                        }
                        .padding(.bottom, 50)
                        
                        // Login form card
                        VStack(spacing: 24) {
                            // Email field
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Email")
                                    .font(.system(size: 16, weight: .semibold))
                                    .foregroundColor(textColor)
                                
                                HStack {
                                    Image(systemName: "envelope.fill")
                                        .foregroundColor(accentColor)
                                        .frame(width: 20)
                                    
                                    TextField("Enter your email", text: $email)
                                        .textFieldStyle(PlainTextFieldStyle())
                                        .font(.system(size: 16))
                                        .foregroundColor(.white)
                                        .keyboardType(.emailAddress)
                                        .autocapitalization(.none)
                                        .disableAutocorrection(true)
                                }
                                .padding(.horizontal, 16)
                                .padding(.vertical, 14)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(cardBackground)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(accentColor.opacity(0.3), lineWidth: 1)
                                        )
                                )
                            }
                            
                            // Password field
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Password")
                                    .font(.system(size: 16, weight: .semibold))
                                    .foregroundColor(textColor)
                                
                                HStack {
                                    Image(systemName: "lock.fill")
                                        .foregroundColor(accentColor)
                                        .frame(width: 20)
                                    
                                    if isPasswordVisible {
                                        TextField("Enter your password", text: $password)
                                            .textFieldStyle(PlainTextFieldStyle())
                                            .font(.system(size: 16))
                                            .foregroundColor(.white)
                                    } else {
                                        SecureField("Enter your password", text: $password)
                                            .textFieldStyle(PlainTextFieldStyle())
                                            .font(.system(size: 16))
                                            .foregroundColor(.white)
                                    }
                                    
                                    Button(action: {
                                        isPasswordVisible.toggle()
                                    }) {
                                        Image(systemName: isPasswordVisible ? "eye.slash.fill" : "eye.fill")
                                            .foregroundColor(accentColor)
                                    }
                                }
                                .padding(.horizontal, 16)
                                .padding(.vertical, 14)
                                .background(
                                    RoundedRectangle(cornerRadius: 12)
                                        .fill(cardBackground)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 12)
                                                .stroke(accentColor.opacity(0.3), lineWidth: 1)
                                        )
                                )
                            }
                            
                            // Forgot password
                            HStack {
                                Spacer()
                                Button("Forgot Password?") {
                                    // Handle forgot password
                                }
                                .font(.system(size: 14, weight: .medium))
                                .foregroundColor(accentColor)
                            }
                            
                            // Login button
                            Button(action: {
                                handleLogin()
                            }) {
                                HStack {
                                    if isLoading {
                                        ProgressView()
                                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                            .scaleEffect(0.8)
                                    } else {
                                        Text("Sign In")
                                            .font(.system(size: 18, weight: .bold))
                                    }
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(
                                    LinearGradient(
                                        gradient: Gradient(colors: [accentColor, Color(hex: "E85D04")]),
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .foregroundColor(.white)
                                .cornerRadius(12)
                                .shadow(color: accentColor.opacity(0.3), radius: 10, x: 0, y: 5)
                            }
                            .disabled(isLoading)
                            
                            // Sign up link
                            HStack {
                                Text("Don't have an account?")
                                    .font(.system(size: 14))
                                    .foregroundColor(secondaryTextColor)
                                
                                Button("Sign Up") {
                                    // Handle sign up
                                }
                                .font(.system(size: 14, weight: .bold))
                                .foregroundColor(textColor)
                            }
                        }
                        .padding(.horizontal, 32)
                        .padding(.vertical, 32)
                        .background(
                            RoundedRectangle(cornerRadius: 24)
                                .fill(cardBackground.opacity(0.8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 24)
                                        .stroke(
                                            LinearGradient(
                                                gradient: Gradient(colors: [accentColor.opacity(0.3), Color(hex: "E85D04").opacity(0.1)]),
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            ),
                                            lineWidth: 1
                                        )
                                )
                        )
                        .padding(.horizontal, 24)
                        .shadow(color: .black.opacity(0.2), radius: 20, x: 0, y: 10)
                        
                        Spacer(minLength: 50)
                    }
                }
            }
        }
        .alert("Login Error", isPresented: $showAlert) {
            Button("OK") { }
        } message: {
            Text(alertMessage)
        }
    }
    
    private func handleLogin() {
        guard !email.isEmpty && !password.isEmpty else {
            alertMessage = "Please fill in all fields"
            showAlert = true
            return
        }
        
        guard email.contains("@") else {
            alertMessage = "Please enter a valid email address"
            showAlert = true
            return
        }
        
        isLoading = true
        
        // Simulate login process
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            isLoading = false
            // Handle successful login here
            print("Login successful for: \(email)")
        }
    }
}

struct LoginView_Previews: PreviewProvider {
    static var previews: some View {
        LoginView()
    }
}
