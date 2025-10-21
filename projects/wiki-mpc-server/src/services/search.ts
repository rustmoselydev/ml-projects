import {
  USER_AGENT,
  WIKIPEDIA_PAGE_BASE,
  WIKIPEDIA_SEARCH_BASE,
} from "../consts/mcp.js";
import { compile } from "html-to-text";
import { JSDOM } from "jsdom";

export async function searchWikipedia(searchTerm: string) {
  const headers = {
    "User-Agent": USER_AGENT,
  };
  try {
    // Wikipedia pages API for the search term, top 5 results
    const pages = await fetch(`${WIKIPEDIA_SEARCH_BASE}${searchTerm}&limit=5`, {
      headers,
    });
    if (!pages.ok) {
      throw new Error(`HTTP error! status: ${pages.status}`);
    }
    const pagesData = await pages.json();
    const pagesPromises: any[] = [];
    pagesData.pages.forEach((page: any) => {
      // Get the actual pages of the returned results
      pagesPromises.push(fetch(`${WIKIPEDIA_PAGE_BASE}${page.key}`));
    });
    const pagesFetched = await Promise.all(pagesPromises);
    const textPromises: any[] = [];
    pagesFetched.forEach((page) => {
      // Raw HTML text conversion
      textPromises.push(page.text());
    });
    const pageText = await Promise.all(textPromises);
    const bodies: any[] = [];
    pageText.forEach((rawText) => {
      // Convert it into a transversable DOM
      const body = new JSDOM(rawText);
      bodies.push(
        // Get the main content
        body.window.document.querySelector(".mw-content-ltr")?.innerHTML
      );
    });
    // Converts the HTML into human readable text
    const compileFunc = compile();
    const finalTexts = bodies.map(compileFunc);
    // This might not be necessary, but it does provide separation between articles
    return finalTexts.join("/////!!!!!");
  } catch (error) {
    console.error("Error making wikipedia request:", error);
    return null;
  }
}
