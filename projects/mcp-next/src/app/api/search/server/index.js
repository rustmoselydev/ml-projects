import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { searchWikipedia } from "./services/search.js";
// Create server instance
const server = new McpServer({
    name: "wikisearch",
    version: "1.0.0",
    capabilities: {
        resources: {},
        tools: {},
    },
});
// Wikipedia search tool
server.tool("search_wikipedia", "Search wikipedia for the information the user wants to know about", {
    searchTerm: z.string().describe(
    // This prompt may need improvement
    "The search term most broadly suited for searching wikipedia for what information the user wants. For example if the user asks for a specific fact about pizza (e.g. different types of pizza, types of cheeses, ingredients, etc), you'd just search for Pizza."),
}, async ({ searchTerm }) => {
    console.error(searchTerm);
    const results = await searchWikipedia(searchTerm);
    console.error(results);
    if (!results) {
        return {
            content: [
                {
                    type: "text",
                    text: "Failed to retrieve wikipedia data",
                },
            ],
        };
    }
    return {
        content: [
            {
                type: "text",
                text: results,
            },
        ],
    };
});
async function main() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("Wikipedia search Server running on stdio");
}
main().catch((error) => {
    console.error("Fatal error in main():", error);
    process.exit(1);
});
