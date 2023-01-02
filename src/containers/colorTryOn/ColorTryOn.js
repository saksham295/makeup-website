import { Button } from "@mui/material";
import React, { useState } from "react";
import "./colorTryOn.css";

const hairColors = [
  "#aa8866",
  "#debe99",
  "#241c11",
  "#4f1a00",
  "#9a3300",
  "#5a3214",
  "#505050",
  "#11642f",
  "#c2a070",
  "#b68a67",
];
const lipColors = [
  "#840016",
  "#3A4763",
  "#D53763",
  "#CC0000",
  "#BF8445",
  "#DAA074",
  "#C0428A",
  "#642209",
  "#E55E58",
  "#EA6770",
];

const ColorTryOn = () => {
  const [file, setFile] = useState();
  const [fileURL, setFileURL] = useState();
  const [apiSuccess, setApiSuccess] = useState(false);
  const [hairColor, setHairColor] = useState();
  const [lipColor, setLipColor] = useState();

  const handleChange = (e) => {
    setFile(e.target.files[0]);
    setFileURL(URL.createObjectURL(e.target.files[0]));
  };

  const handleHairColor = (i) => {
    setHairColor(hairColors[i]);
  };

  const handleLipColor = (j) => {
    setLipColor(lipColors[j]);
  };

  const handleSubmission = async () => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("haircolor", hairColor);
    console.log(
      "🚀 ~ file: ColorTryOn.js:54 ~ handleSubmission ~ hairColor",
      hairColor
    );
    formData.append("lipcolor", lipColor);
    console.log(
      "🚀 ~ file: ColorTryOn.js:55 ~ handleSubmission ~ lipColor",
      lipColor
    );

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
      <h1>Upload Your Image:</h1>
      <input type="file" onChange={handleChange} />
      <h1>Select Hair Color:</h1>
      <div className="colorList">
        {hairColors.map((color, i) => (
          <div
            key={i}
            style={{
              background: color,
              border:
                hairColor === hairColors[i]
                  ? "3px solid var(--color-secondary)"
                  : "",
            }}
            onClick={() => handleHairColor(i)}
          ></div>
        ))}
      </div>
      <h1>Select Lipstick Color:</h1>
      <div className="colorList">
        {lipColors.map((color, j) => (
          <div
            key={j}
            style={{
              background: color,
              border:
                lipColor === lipColors[j]
                  ? "3px solid var(--color-secondary)"
                  : "",
            }}
            onClick={() => handleLipColor(j)}
          ></div>
        ))}
      </div>

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
