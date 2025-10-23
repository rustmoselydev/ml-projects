import { NextResponse } from "next/server";
import { MCPClient } from "./client";
import path from "path";

// Path to our compiled server file
const serverPath = path.join(
  process.cwd(),
  "src",
  "app",
  "api",
  "search",
  "server",
  "index.js"
);

// Node.js REST endpoint for FE
export async function POST(req: Request) {
  try {
    const mcpClient = new MCPClient();
    await mcpClient.connectToServer(serverPath);
    const data = await req.json();
    const res = await mcpClient.processQuery(
      `Please search wikipedia and find out: ${data.query}`
    );
    return NextResponse.json({ result: res });
  } catch (e: any) {
    return NextResponse.json(
      { error: e.message || "Internal server error" },
      { status: 500 }
    );
  }
}
