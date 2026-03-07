
  import { createRoot } from "react-dom/client";
  import { SplashScreen } from "@capacitor/splash-screen";
  import App from "./App.tsx";
  import "./index.css";

  createRoot(document.getElementById("root")!).render(<App />);
  SplashScreen.hide();
  