"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";

interface ChatItem {
  source: string;
  text: string;
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [chat, setChat] = useState<ChatItem[]>([]);

  const submitQuery = async () => {
    const newChat = chat.concat({
      source: "User",
      text: query,
    });
    setChat(newChat);
    setQuery("");
    const resp = await fetch(window.location.origin + "/api/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: query }),
    });
    const data = await resp.json();
    const respChat = newChat.concat({
      source: "AI",
      text: data.result,
    });
    setChat(respChat);
  };

  return (
    <div className="w-full h-[100vh] relative">
      <div className="w-full p-4">
        {chat.map((val) => {
          return (
            <div
              className={`mb-4 max-w-1/2 w-fit rounded p-2 ${
                val.source === "User" ? "ml-auto bg-blue-200" : "bg-gray-200"
              }`}
              key={val.text}
            >
              <ReactMarkdown>{val.text}</ReactMarkdown>
            </div>
          );
        })}
      </div>
      <div className="absolute left-0 bottom-0 w-full p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submitQuery();
          }}
        >
          <input
            type="text"
            className="w-full border border-black rounded"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
            }}
          />
        </form>
      </div>
    </div>
  );
}
