import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangChainAgentAdapter } from "langchain-copilotkit";
import { createSlideAgent } from "../../../agent";

const agent = createSlideAgent();

const runtime = new CopilotRuntime({
  agents: {
    default: new LangChainAgentAdapter({
      agent,
      stateKeys: ["files"],
    }),
  },
});

export const { handleRequest: POST } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  endpoint: "/api/copilotkit",
});
