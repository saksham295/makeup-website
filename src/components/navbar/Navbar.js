import {
  AppBar,
  Toolbar,
  Button,
  IconButton,
  Drawer,
  Link,
  MenuItem,
} from "@material-ui/core";
import "./navbar.css";
import MenuIcon from "@mui/icons-material/Menu";
import React, { useState, useEffect } from "react";
import { Link as RouterLink } from "react-router-dom";
import logo from "../../assets/logo.png";

const headersData = [
  {
    label: "Home",
    href: "/",
  },
  {
    label: "Features",
    href: "/features",
  },
  {
    label: "Products",
    href: "/products",
  },
  {
    label: "Team",
    href: "/team",
  },
  {
    label: "Healthcare",
    href: "/healthcare",
  },
];

const Navbar = () => {
  const [state, setState] = useState({
    mobileView: false,
    drawerOpen: false,
  });

  const { mobileView, drawerOpen } = state;

  useEffect(() => {
    const setResponsiveness = () => {
      return window.innerWidth <= 950
        ? setState((prevState) => ({ ...prevState, mobileView: true }))
        : setState((prevState) => ({ ...prevState, mobileView: false }));
    };

    setResponsiveness();

    window.addEventListener("resize", () => setResponsiveness());

    return () => {
      window.removeEventListener("resize", () => setResponsiveness());
    };
  }, []);

  const displayDesktop = () => {
    return (
      <Toolbar className="toolbar">
        {deepakshiGlobalLogo}
        <div
          style={{
            width: "50%",
            display: "flex",
            justifyContent: "space-evenly",
          }}
        >
          {getMenuButtons()}
        </div>
        {getTryNowButton}
      </Toolbar>
    );
  };

  const displayMobile = () => {
    const handleDrawerOpen = () =>
      setState((prevState) => ({ ...prevState, drawerOpen: true }));
    const handleDrawerClose = () =>
      setState((prevState) => ({ ...prevState, drawerOpen: false }));

    return (
      <Toolbar className="toolbar">
        <IconButton
          {...{
            edge: "start",
            color: "inherit",
            "aria-label": "menu",
            "aria-haspopup": "true",
            onClick: handleDrawerOpen,
          }}
        >
          <MenuIcon style={{ color: "white" }} />
        </IconButton>

        <Drawer
          {...{
            anchor: "left",
            open: drawerOpen,
            onClose: handleDrawerClose,
          }}
        >
          <div className="drawerContainer">{getDrawerChoices()}</div>
        </Drawer>

        <div>{deepakshiGlobalLogo}</div>
        {getTryNowButton}
      </Toolbar>
    );
  };

  const getDrawerChoices = () => {
    return headersData.map(({ label, href }) => {
      return (
        <Link
          {...{
            component: RouterLink,
            to: href,
            color: "inherit",
            style: {
              textDecoration: "none",
            },
            key: label,
          }}
        >
          <MenuItem
            style={{
              fontFamily: "var(--font-family)",
              fontWeight: 700,
              color: "white",
            }}
          >
            {label}
          </MenuItem>
        </Link>
      );
    });
  };

  const deepakshiGlobalLogo = (
    <div className="logoContainer">
      <img src={logo} alt="logo" style={{ height: "9vh", width: "100%" }} />
    </div>
  );

  const getMenuButtons = () => {
    return headersData.map(({ label, href }) => {
      return (
        <Button
          style={{ color: "white" }}
          {...{
            key: label,
            color: "inherit",
            to: href,
            component: RouterLink,
            className: "menuButton",
          }}
        >
          {label}
        </Button>
      );
    });
  };

  const getTryNowButton = <Button className="tryNowButton">TRY NOW!</Button>;

  return (
    <AppBar className="navbar">
      {mobileView ? displayMobile() : displayDesktop()}
    </AppBar>
  );
};

export default Navbar;
