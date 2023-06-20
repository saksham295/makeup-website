import { Button, CircularProgress } from "@mui/material";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { useState } from "react";
import "./faceTryOn.css";

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

const FaceTryOn = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [file1URL, setFile1URL] = useState(null);
  const [file2URL, setFile2URL] = useState(null);
  const [outputImage, setOutputImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  const handleFile1Change = (e) => {
    setOutputImage(null);
    setError(null);
    setSuccessMessage(null);
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > MAX_FILE_SIZE) {
        setError(
          "File size exceeds the limit (5MB). Please choose a smaller file."
        );
        return;
      }
      if (!selectedFile.type.startsWith("image/")) {
        setError(
          "Invalid file format. Please choose an image file (JPEG, PNG, etc.)."
        );
        return;
      }
      setFile1(selectedFile);
      setFile1URL(URL.createObjectURL(selectedFile));
    }
  };

  const handleFile2Change = (e) => {
    setOutputImage(null);
    setError(null);
    setSuccessMessage(null);
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > MAX_FILE_SIZE) {
        setError(
          "File size exceeds the limit (5MB). Please choose a smaller file."
        );
        return;
      }
      if (!selectedFile.type.startsWith("image/")) {
        setError(
          "Invalid file format. Please choose an image file (JPEG, PNG, etc.)."
        );
        return;
      }
      setFile2(selectedFile);
      setFile2URL(URL.createObjectURL(selectedFile));
    }
  };

  const handleSubmission = async () => {
    if (!file1 || !file2) {
      return;
    }

    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);

    const formData = new FormData();
    formData.append("makeup", file1);
    formData.append("no_makeup", file2);

    try {
      const response = await fetch("http://127.0.0.1:7001/pairwise", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("An error occurred while processing the request.");
      }

      if (response.ok) {
        const blob = await response.blob();
        const imageURL = URL.createObjectURL(blob);
        setOutputImage(imageURL);
        setSuccessMessage("Image transfer completed successfully!");
        setFile1(null);
        setFile1URL(null);
        setFile2(null);
        setFile2URL(null);
      } else {
        console.error("Error:", response.status);
        setError("An error occurred. Please try again later.");
      }
    } catch (error) {
      setError("An error occurred. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="face">
      <h1>
        Please Upload The Image Whose Makeup You Want To Transfer On Your Image:
      </h1>
      <div className="file-input" style={{ marginBottom: "2rem" }}>
        <input
          type="file"
          onChange={handleFile1Change}
          id="file1"
          className="inputfile"
          accept="image/*"
        />
        <label htmlFor="file1">Choose File</label>
      </div>
      <h1>Please Upload Your Image Without Makeup:</h1>
      <div className="file-input">
        <input
          type="file"
          onChange={handleFile2Change}
          id="file2"
          className="inputfile"
          accept="image/*"
        />
        <label htmlFor="file2">Choose File</label>
      </div>
      <div className="face-transfer">
        {file1URL && !outputImage && !error && (
          <>
            <img src={file1URL} alt="input 1" />
            {file2URL && (
              <>
                <Button
                  onClick={handleSubmission}
                  disabled={!file2 || isLoading}
                >
                  {isLoading ? (
                    <CircularProgress size={24} style={{ color: "white" }} />
                  ) : (
                    <>
                      {"Start Transfer  "}
                      <ArrowForwardIcon />
                    </>
                  )}
                </Button>
                <img src={file2URL} alt="input 2" />
              </>
            )}
          </>
        )}
        {error && <p className="error-message">{error}</p>}
      </div>
      {outputImage && (
        <div className="output-container">
          <img src={outputImage} alt="output" />
          {successMessage && <p>{successMessage}</p>}
        </div>
      )}
    </div>
  );
};

export default FaceTryOn;
