"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import streamifier from "streamifier";

export default function Home() {
  const [image, setImage] = useState<any>();
  const [prediction, setPrediction] = useState<any>();

  const toBase64 = (file: any) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
    });

  const submit = async (e: any) => {
    e.preventDefault();
    const formData = new FormData();
    const img: any = await toBase64(image);
    formData.append("image", img);
    const req = await fetch("/api/guitar", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: img }),
    });
    const result = await req.json();
    setPrediction(result);
  };

  const imageChanged = (e: any) => {
    if (e.target.files) {
      const file = e.target.files[0];
      setImage(file);
    }
  };

  return (
    <main>
      <form onSubmit={submit}>
        <input name="image" type="file" onChange={imageChanged} />
        <button role="submit">Submit</button>
      </form>
      {prediction && (
        <div>
          <div className="capitalize">Prediction: {prediction?.result}</div>
          <div>
            Confidence:{" "}
            {(parseFloat(prediction?.confidence) * 100).toFixed(0) + "%"}
          </div>
        </div>
      )}
    </main>
  );
}
