FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

WORKDIR /app

RUN printf '%s\n' '{"service":"demo-api","enabled":false,"retries":0,"endpoints":{"health":"/status"}}' > /app/service-config.json
