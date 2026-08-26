import { OpenRouter } from "@openrouter/sdk";

import { getOpenRouterConfig } from "./openrouter";

const { apiKey, model, provider, routingProfile } = getOpenRouterConfig();
const providerPreferences = {
  ...("sort" in provider ? { sort: provider.sort } : {}),
  requireParameters: provider.require_parameters,
};

const openRouter = new OpenRouter({ apiKey });
const result = await openRouter.chat.send({
  xOpenRouterMetadata: "enabled",
  chatRequest: {
    model,
    messages: [{ role: "user", content: "Reply with only OK." }],
    provider: providerPreferences,
    stream: false,
  },
});

if (!("choices" in result)) {
  throw new Error("OpenRouter returned an unexpected streaming response");
}

const metadata = result.openrouterMetadata;
if (!metadata) {
  throw new Error("OpenRouter response did not include Router Metadata");
}

console.log(
  JSON.stringify(
    {
      model,
      routingProfile,
      candidateProviders: metadata.endpoints.available.map((endpoint) => ({
        provider: endpoint.provider,
        selected: endpoint.selected,
      })),
      selectedProvider: metadata.endpoints.available.find(
        (endpoint) => endpoint.selected,
      )?.provider,
      attempts: metadata.attempts ?? [],
      attempt: metadata.attempt,
      fallbackOccurred: metadata.attempt > 1,
    },
    null,
    2,
  ),
);
