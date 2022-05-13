import makeup from "../../assets/makeup.jpeg";
import { makeStyles } from "@material-ui/core";
import Features from "./Features";

const useStyles = makeStyles(() => ({
  makeupimg: {
    width: "100%",
    // margin: "5px 0px",
    // height: "80vh",
    // "@media (max-width: 1000px)": {
    //   height: "40vh",
    // },
  },
}));

const Home = () => {
  const { makeupimg } = useStyles();
  return (
    <div>
      <img className={makeupimg} src={makeup} alt="makeup"></img>
      <Features />
    </div>
  );
};

export default Home;
