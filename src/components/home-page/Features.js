import * as React from "react";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import feature1 from "../../assets/feature1.jpeg";
import { makeStyles } from "@material-ui/core";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { Link } from "react-router-dom";

const useStyles = makeStyles(() => ({
  features_container: {
    display: "flex",
    flexDirection: "row",
    justifyContent: "center",
    "@media (max-width: 800px)": {
      flexDirection: "column",
    },
  },
  card: {
    display: "flex",
    margin: "30px 10px",
    flexWrap: "wrap",
    justifyContent: "center",
    // width: "30%",
    "@media (max-width: 1250px )": {
      height: "80vh",
      margin: "15px 12px",
    },
    "@media (max-width: 375px )": {
      height: "120vh",
      margin: "10px 10px",
    },
    // "@media (max-width: 800px)": {
    //   width: "100%",
    // },
  },
}));

const features = [
  {
    img: feature1,
    name: "Cosmetic recommendation",
    desc: "Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime quasi modi, similique praesentium deleniti eligendi est, unde quaerat aspernatur tempore totam vel qui ad mollitia tenetur rerum nemo, eaque non?",
    link: "/cosmetic",
  },
  {
    img: feature1,
    name: "Acne Detection and Cures",
    desc: "Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime quasi modi, similique praesentium deleniti eligendi est, unde quaerat aspernatur tempore totam vel qui ad mollitia tenetur rerum nemo, eaque non?",
    link: "/acne-detection",
  },
  {
    img: feature1,
    name: "Makeup recommendations on the basis of occasion",
    desc: "Lorem ipsum dolor sit amet consectetur adipisicing elit. Maxime quasi modi, similique praesentium deleniti eligendi est, unde quaerat aspernatur tempore totam vel qui ad mollitia tenetur rerum nemo, eaque non?",
    link: "/makeup",
  },
];

const Features = () => {
  const { card, features_container } = useStyles();
  return (
    // <>
    //   <div style={{ display: "flex" }}>
    //     <Typography
    //       variant="h3"
    //       align="center"
    //       style={{
    //         margin: 5,
    //         padding: 10,
    //         fontSize: 38,
    //         fontWeight: "bolder",
    //         fontFamily: "Source Sans Pro, sans-serif",
    //         color: "#6f3e29",
    //         backgroundColor: "rgb(243 208 199)",
    //         borderRadius: "0px 30px",
    //         boxShadow: "grey 15px 15px 8px",
    //         width: "50%",
    //       }}
    //     >
    //       Features
    //     </Typography>
    //   </div>
    <div className={features_container}>
      {features.map((feature, i) => (
        <Card
          sx={{
            background: "rgb(243 208 199)",
            boxShadow: "lightgrey 20px 20px 30px",
            borderRadius: "20px",
          }}
          className={card}
          key={i}
        >
          <CardMedia
            component="img"
            height="50%"
            image={feature.img}
            alt="feature"
          />
          <CardContent
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <Typography
              gutterBottom
              variant="h5"
              align="center"
              component="div"
              color="#6f3e29"
              style={{
                fontFamily: "Source Sans Pro, sans-serif",
                fontWeight: "bold",
              }}
            >
              {feature.name}
            </Typography>
            <Typography
              variant="body2"
              color="#6f3e29"
              align="justify"
              style={{
                fontFamily: "Source Sans Pro, sans-serif",
                fontWeight: "bold",
              }}
            >
              {feature.desc}
            </Typography>
          </CardContent>
          <CardActions>
            <Link to={feature.link}>
              <Button size="medium" style={{ color: "#6f3e29" }}>
                <ArrowForwardIcon />
              </Button>
            </Link>
          </CardActions>
        </Card>
      ))}
    </div>
    // </>
  );
};

export default Features;
