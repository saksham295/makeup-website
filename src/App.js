import "./App.css";
import { useRoutes, BrowserRouter as Router } from "react-router-dom";
import Navbar from "./components/navbar/Navbar";
import Home from "./containers/home/Home";
import AcneDetection from "./containers/acne/AcneDetection";
import { Helmet, HelmetProvider } from "react-helmet-async";
import ColorTryOn from "./containers/colorTryOn/ColorTryOn";

function App() {
  let routes = useRoutes([
    {
      path: "/",
      element: <Home />,
    },
    {
      path: "/acne-detection",
      element: <AcneDetection />,
    },
    {
      path: "/color-tryon",
      element: <ColorTryOn />,
    },
  ]);
  return routes;
}

const AppWrapper = () => {
  return (
    <HelmetProvider>
      <Helmet>
        <title>Encapture Beauty</title>
      </Helmet>
      <Router>
        <Navbar />
        <App />
      </Router>
    </HelmetProvider>
  );
};

export default AppWrapper;
