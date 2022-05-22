import {
  AppBar,
  Toolbar,
  Typography,
  makeStyles,
  Button,
  IconButton,
  Drawer,
  Link,
  MenuItem,
} from "@material-ui/core";
import MenuIcon from "@mui/icons-material/Menu";
import React, { useState, useEffect } from "react";
import { Link as RouterLink } from "react-router-dom";
import dglogo from "../../assets/logo1.jpeg";

const headersData = [
  {
    label: "Home",
    href: "/",
  },
  {
    label: "Mentors",
    href: "/mentors",
  },
  {
    label: "About",
    href: "/about-us",
  },
  {
    label: "Contact Us",
    href: "/contact-us",
  },
];

const useStyles = makeStyles(() => ({
  header: {
    position: "relative",
    backgroundColor: "#fafafa",
    paddingRight: "79px",
    paddingLeft: "118px",
    "@media (max-width: 950px)": {
      paddingLeft: 0,
      paddingRight: 0,
    },
  },
  logo: {
    fontFamily: "Source Sans Pro, sans-serif",
    fontWeight: 600,
    margin: "0px 5px",
    color: "#000",
    textAlign: "left",
  },
  logoContainer: {
    display: "flex",
    "@media (max-width: 950px)": {
      width: "100%",
    },
  },
  menuButton: {
    fontFamily: "Source Sans Pro, sans-serif",
    fontWeight: 700,
    size: "18px",
    marginLeft: "38px",
  },
  toolbar: {
    display: "flex",
    justifyContent: "space-between",
  },
  drawerContainer: {
    padding: "20px 30px",
  },
}));

const Navbar = () => {
  const { header, logo, menuButton, toolbar, drawerContainer, logoContainer } =
    useStyles();

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
      <Toolbar className={toolbar}>
        {deepakshiGlobalLogo}
        <div>{getMenuButtons()}</div>
      </Toolbar>
    );
  };

  const displayMobile = () => {
    const handleDrawerOpen = () =>
      setState((prevState) => ({ ...prevState, drawerOpen: true }));
    const handleDrawerClose = () =>
      setState((prevState) => ({ ...prevState, drawerOpen: false }));

    return (
      <Toolbar>
        <IconButton
          {...{
            edge: "start",
            color: "inherit",
            "aria-label": "menu",
            "aria-haspopup": "true",
            onClick: handleDrawerOpen,
          }}
        >
          <MenuIcon style={{ color: "#000" }} />
        </IconButton>

        <Drawer
          {...{
            anchor: "left",
            open: drawerOpen,
            onClose: handleDrawerClose,
          }}
        >
          <div className={drawerContainer}>{getDrawerChoices()}</div>
        </Drawer>

        <div>{deepakshiGlobalLogo}</div>
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
              fontFamily: "Source Sans Pro, sans-serif",
              fontWeight: 700,
            }}
          >
            {label}
          </MenuItem>
        </Link>
      );
    });
  };

  const deepakshiGlobalLogo = (
    <div className={logoContainer}>
      <img src={dglogo} alt="logo" height="30" style={{ margin: "2px" }} />
      <Typography variant="h6" component="h1" className={logo}>
        Deepakshi Global
      </Typography>
    </div>
  );

  const getMenuButtons = () => {
    return headersData.map(({ label, href }) => {
      return (
        <Button
          style={{ color: "#000" }}
          {...{
            key: label,
            color: "inherit",
            to: href,
            component: RouterLink,
            className: menuButton,
          }}
        >
          {label}
        </Button>
      );
    });
  };

  return (
    <header>
      <AppBar className={header}>
        {mobileView ? displayMobile() : displayDesktop()}
      </AppBar>
    </header>
  );
};

export default Navbar;
