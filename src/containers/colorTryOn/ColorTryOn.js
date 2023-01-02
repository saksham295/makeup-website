import { Button } from "@mui/material";
import React, { useState } from "react";
import "./colorTryOn.css";

const ColorTryOn = () => {
  const [file, setFile] = useState();
  const [fileURL, setFileURL] = useState();
  const [apiSuccess, setApiSuccess] = useState(false);

  const handleChange = (e) => {
    setFile(e.target.files[0]);
    setFileURL(URL.createObjectURL(e.target.files[0]));
  };

  const handleSubmission = async () => {
    const formData = new FormData();
    formData.append("file", file);

    await fetch("http://192.168.0.104:7001/display", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((result) => {
        console.log(result);
        setApiSuccess(true);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  return (
    <div className="color">
      <h1>Please Upload Your Image:</h1>
      <input type="file" onChange={handleChange} />
      <div className="colorTryOn">
        {file !== undefined ? (
          <>
            <img src={fileURL} alt="input" />
            <Button onClick={handleSubmission}>Try Colors</Button>
          </>
        ) : (
          <></>
        )}
        {apiSuccess === true ? (
          <div className="colorOutput">
            <img src={require("../../assets/colorOutput.png")} alt="output" />
          </div>
        ) : (
          <></>
        )}
      </div>
    </div>
  );
};

export default ColorTryOn;
