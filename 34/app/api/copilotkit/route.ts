import {
  CopilotRuntime,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangChainAgentAdapter } from "langchain-copilotkit";
import { createAcpSupervisorAgent } from "../../../agent";

const agent = createAcpSupervisorAgent();

const runtime = new CopilotRuntime({
  agents: {
    default: new LangChainAgentAdapter({
      agent,
    }),
  },
});

export const { handleRequest: POST } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  endpoint: "/api/copilotkit",
});
