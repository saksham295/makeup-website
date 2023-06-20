import "./home.css";
import makeup from "../../assets/makeup.png";
import { Box, Button, Container, Divider, Typography } from "@mui/material";
import getStarted1 from "../../assets/getstarted1.png";
import getStarted2 from "../../assets/getstarted2.png";
import feature1 from "../../assets/feature1.png";
import feature2 from "../../assets/feature2.png";
import feature3 from "../../assets/feature3.png";
import thanksIcon from "../../assets/thanks.svg";
import { Link } from "react-router-dom";

const Home = () => {
  return (
    <>
      <div className="header">
        <div className="headerContent">
          <h1>
            TRY ON YOUR{" "}
            <span style={{ color: "var(--color-secondary)" }}>FAVORITE</span>{" "}
            LOOKS
          </h1>
          <p>
            VIRTUALLY TRY ON YOUR FAVORITE LOOKS & PRODUCTS USING OUR VIRTUAL
            TRY-ON TOOL. BROWSE, PLAY, AND SEE HOW THE SHADES LOOK ON YOU.
          </p>
        </div>
        <div className="headerImg">
          <img src={makeup} alt="makeup" />
        </div>
      </div>

      <Container maxWidth="lg">
        <Box>
          <Typography
            style={{
              fontFamily: "var(--font-family)",
              fontWeight: 700,
              fontSize: 24,
              margin: "1rem 0",
            }}
          >
            GET STARTED IN A SNAP!
          </Typography>
          <div className="getStarted">
            <div className="getStarted1">
              <img src={getStarted1} alt="Get Started 1" />
              <p>
                Try on a product with live camera mode, upload a front-facing
                photo of yourself, or choose a model with similar features to
                yours.
              </p>
            </div>
            <div className="getStarted2">
              <p>
                Then browse and select all the shades from our catalog to try on
                virtually! <br /> <br /> Use the before/after button to compare.
              </p>
              <img src={getStarted2} alt="Get Started 2" />
            </div>
          </div>

          <Divider />

          <div className="feature">
            <div className="featureImg">
              <img src={feature1} alt="feature1" />
            </div>
            <div className="featureContent">
              <h1>LIMITLESS SHADES AND FINISHES</h1>
              <p>
                Experiment new finishes and play with colors to find the right
                match for you in just a few clicks.
              </p>
              <Link
                to="color-tryon"
                style={{ textDecoration: "none", display: "flex" }}
              >
                <Button>Try It On</Button>
              </Link>
            </div>
          </div>

          <Divider />

          <div className="feature columnReverse">
            <div className="featureContent">
              <h1>NEW! FACE TRY-ON</h1>
              <p>
                Match yourself with the perfect shades and coverage of
                foundations, concealers, powders, blushes, bronzers, and setting
                sprays. You can even compare the finishes – from matte to satin,
                to radiant all with amazing accuracy!
              </p>
              <div>
                <Link
                  to="face-tryon"
                  style={{ textDecoration: "none", display: "flex" }}
                >
                  <Button>Try It On</Button>
                </Link>
              </div>
            </div>
            <div className="featureImg">
              <img src={feature2} alt="feature2" />
            </div>
          </div>

          <Divider />

          <div className="feature">
            <div className="featureImg">
              <img src={feature3} alt="feature3" />
            </div>
            <div className="featureContent">
              <h1>ACNE DETECTION</h1>
              <p>
                In the mood for epic artistry or a work-from-home glow? Pick
                your favorite looks from our looks catalog, try it on, and shop!
              </p>
              <Link
                to="acne-detection"
                style={{ textDecoration: "none", display: "flex" }}
              >
                <Button>Try It On</Button>
              </Link>
            </div>
          </div>

          <div className="thanks">
            <img src={thanksIcon} alt="Thank You" />
            <h1>WE KNOW YOU'LL LOVE YOUR MAKEUP</h1>
          </div>
        </Box>
      </Container>

      <div className="footer"></div>
    </>
  );
};

export default Home;
