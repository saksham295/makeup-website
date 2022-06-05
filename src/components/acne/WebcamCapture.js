import React, { useState } from "react";
import Button from "@mui/material/Button";
import Webcam from "react-webcam";
import { makeStyles } from "@material-ui/core";

const videoConstraints = {
  facingMode: "user",
};

const useStyles = makeStyles(() => ({
  button: {
    width: "50%",
    backgroundColor: "rgb(243 208 199) !important",
    color: "#6f3e29 !important",
    fontWeight: "bold !important",
    // "@media (max-width: 950px)": {
    //   paddingLeft: 0,
    //   paddingRight: 0,
    // },
  },
  webcam: {
    width: "40%",
    marginTop: "10px",
    margin: "5px",
    "@media (max-width: 800px)": {
      margin: 0,
      width: "100%",
    },
  },
  captureButton: {
    width: "40%",
    display: "flex",
    justifyContent: "center",
    "@media (max-width: 800px)": {
      width: "100%",
    },
  },
}));

export const WebcamCapture = () => {
  const { button, webcam, captureButton } = useStyles();
  const [image, setImage] = useState("");
  const webcamRef = React.useRef(null);

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImage(imageSrc);
  }, []);

  return (
    <div>
      <div style={{ display: "flex" }}>
        {image === "" ? (
          <Webcam
            audio={false}
            mirrored={true}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            className={webcam}
          />
        ) : (
          <img src={image} alt="myimage" className={webcam} />
        )}
        {image !== "" ? (
          <>
            <Button
              variant="contained"
              className={button}
              style={{ width: "20%" }}
            >
              Process?
            </Button>
            <img alt="ProcessedImage" className={webcam} />
          </>
        ) : (
          <></>
        )}
      </div>
      <div className={captureButton}>
        {image !== "" ? (
          <Button
            variant="contained"
            className={button}
            onClick={(e) => {
              e.preventDefault();
              setImage("");
            }}
          >
            Retake Image
          </Button>
        ) : (
          <Button
            variant="contained"
            className={button}
            onClick={(e) => {
              e.preventDefault();
              capture();
            }}
          >
            Capture
          </Button>
        )}
      </div>
    </div>
  );
};

export default WebcamCapture;
