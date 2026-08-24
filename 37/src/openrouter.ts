export const MODEL = "deepseek/deepseek-v4-flash-0731";

const ROUTING_PROFILES = {
  default: { require_parameters: true },
  price: { sort: "price", require_parameters: true },
  latency: { sort: "latency", require_parameters: true },
  throughput: { sort: "throughput", require_parameters: true },
} satisfies Record<
  string,
  {
    sort?: "price" | "latency" | "throughput";
    require_parameters: true;
  }
>;

export function getOpenRouterConfig() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error("OPENROUTER_API_KEY is required");
  }

  const routingProfile = process.env.OPENROUTER_ROUTING_PROFILE ?? "default";
  if (!(routingProfile in ROUTING_PROFILES)) {
    throw new Error(`Unknown routing profile: ${routingProfile}`);
  }

  return {
    apiKey,
    model: MODEL,
    routingProfile,
    provider:
      ROUTING_PROFILES[routingProfile as keyof typeof ROUTING_PROFILES],
  };
}
