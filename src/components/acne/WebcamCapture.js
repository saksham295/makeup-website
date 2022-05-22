import React, { useState } from "react";
import Button from "@mui/material/Button";
import Webcam from "react-webcam";

// const WebcamComponent = () => <Webcam />;

const videoConstraints = {
  facingMode: "user",
  width: "40%",
};

export const WebcamCapture = () => {
  const [image, setImage] = useState("");
  const webcamRef = React.useRef(null);

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImage(imageSrc);
  });

  return (
    <div style={{ width: "50%" }}>
      <div>
        {image === "" ? (
          <Webcam
            audio={false}
            mirrored={true}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
          />
        ) : (
          <img src={image} alt="myimage" />
        )}
      </div>
      <div>
        {image !== "" ? (
          <Button
            variant="contained"
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
