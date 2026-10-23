FROM node:22.23.4-slim AS node-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && sed -i \
        -e '/^mozilla\/AAA_Certificate_Services\.crt$/s/^/!/' \
        -e '/^mozilla\/emSign_ECC_Root_CA_-_G3\.crt$/s/^/!/' \
        /etc/ca-certificates.conf \
    && update-ca-certificates --fresh \
    && rm -rf /var/lib/apt/lists/*

FROM node-base AS base
RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://bun.sh/install | BUN_INSTALL=/usr/local bash -s -- bun-v1.4.4

# --- deps stage ---
FROM base AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install

# --- builder stage ---
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN ANTHROPIC_API_KEY=dummy-build-key bun run build --webpack

# --- runner stage ---
FROM node-base AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

# Copy standalone build
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Copy workspace
COPY --from=builder /app/workspace ./workspace
RUN chown -R nextjs:nodejs /app/workspace

USER nextjs
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
