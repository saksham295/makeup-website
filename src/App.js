import "./App.css";
import { useRoutes, BrowserRouter as Router } from "react-router-dom";
import Navbar from "./components/navbar/Navbar";
import Home from "./components/home-page/Home";
import AcneDetection from "./components/acne/AcneDetection";
// import Footer from "./components/footer/Footer";
import { Helmet, HelmetProvider } from "react-helmet-async";

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
  ]);
  return routes;
}

const AppWrapper = () => {
  return (
    <HelmetProvider>
      <Helmet>
        <title>Deepakshi Global</title>
      </Helmet>
      <Router>
        <Navbar />
        <App />
        {/* <Footer /> */}
      </Router>
    </HelmetProvider>
  );
};

export default AppWrapper;
