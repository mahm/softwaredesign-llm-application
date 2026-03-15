import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: [
    "deepagents",
    "@langchain/core",
    "@langchain/langgraph",
    "@langchain/anthropic",
    "langchain-copilotkit",
  ],
};

export default nextConfig;
