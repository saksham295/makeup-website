import { Button } from "@mui/material";
import React, { useState } from "react";
import "./acneDetection.css";

const AcneDetection = () => {
  const [file, setFile] = useState();
  const [fileURL, setFileURL] = useState();
  const [acneScore, setAcneScore] = useState("");

  const handleChange = (e) => {
    setFile(e.target.files[0]);
    setFileURL(URL.createObjectURL(e.target.files[0]));
    setAcneScore("");
  };

  const handleSubmission = async () => {
    const formData = new FormData();
    formData.append("file", file);

    await fetch("http://192.168.0.104:7002/display", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((result) => {
        console.log(result);
        setAcneScore(result);
      });
    console
      .log(
        "🚀 ~ file: AcneDetection.js:29 ~ .then ~ setAcneScore",
        acneScore.AcneScore
      )
      .catch((error) => {
        console.error("Error:", error);
        setAcneScore("");
      });
  };

  return (
    <div className="acne">
      <h1>Please Upload Your Image:</h1>
      <input type="file" onChange={handleChange} />
      <div className="acneDetection">
        {file !== undefined ? (
          <>
            <img src={fileURL} alt="input" />
            <Button onClick={handleSubmission}>Detect Acne</Button>
          </>
        ) : (
          <></>
        )}
        {acneScore !== "" ? (
          <div className="acneOutput">
            <img src={require("../../assets/output.png")} alt="output" />
            <p>
              Acne Severity Score: {acneScore.AcneScore}{" "}
              {acneScore.AcneScore < 7 ? (
                "(minor)"
              ) : (
                <>{acneScore.AcneScore < 15 ? "(mild)" : "(severe)"}</>
              )}
            </p>
          </div>
        ) : (
          <></>
        )}
      </div>
      <div className="acnePrevention">
        {acneScore !== "" ? (
          <>
            <h2>
              Here are some preventive measures for acne that you can take based
              on the severity of your acne:
            </h2>
            {acneScore.AcneScore < 7 ? (
              <ol>
                <li>
                  Use over-the-counter acne products containing salicylic acid
                  or benzoyl peroxide to help unclog pores and reduce
                  inflammation.
                </li>
                <li>
                  Avoid using harsh or abrasive skin care products that can
                  irritate your skin.
                </li>
                <li>
                  Use non-comedogenic skin care and makeup products to reduce
                  the risk of clogged pores.
                </li>
                <li>
                  Avoid picking or squeezing your acne, as this can lead to more
                  breakouts and scarring.
                </li>
                <li>
                  Drink plenty of water and eat a healthy, balanced diet to help
                  keep your skin healthy.
                </li>
              </ol>
            ) : (
              <>
                {acneScore.AcneScore < 15 ? (
                  <ol>
                    <li>
                      Consider using prescription acne medications such as
                      retinoids or antimicrobials.
                    </li>
                    <li>
                      Avoid using oil-based skin care and makeup products, as
                      they can contribute to clogged pores.
                    </li>
                    <li>
                      Use a gentle, oil-free cleanser to wash your face twice a
                      day.
                    </li>
                    <li>
                      Consider using a light, oil-free moisturizer to keep your
                      skin hydrated without adding excess oil.
                    </li>
                    <li>
                      Consult a dermatologist for personalized treatment
                      recommendations.
                    </li>
                  </ol>
                ) : (
                  <ol>
                    <li>
                      Consider using a combination of prescription acne
                      medications, such as retinoids, antimicrobials, and oral
                      contraceptives.
                    </li>
                    <li>
                      Follow a strict skin care routine to keep your skin clean
                      and clear.
                    </li>
                    <li>
                      Avoid using oil-based skin care and makeup products, and
                      use non-comedogenic products instead.
                    </li>
                    <li>
                      Consult a dermatologist for personalized treatment
                      recommendations, as severe acne may require more intensive
                      treatment.
                    </li>
                  </ol>
                )}
              </>
            )}
          </>
        ) : (
          <></>
        )}
      </div>
    </div>
  );
};

export default AcneDetection;
