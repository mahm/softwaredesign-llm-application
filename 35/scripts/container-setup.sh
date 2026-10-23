#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/deepagents-tbench-autotune-ts

apt-get update
apt-get install -y ca-certificates curl git gzip tar unzip

# Bunの取得にも、侵害された中間CAを信頼しないCAストアを使用する。
sed -i \
  -e '/^mozilla\/AAA_Certificate_Services\.crt$/s/^/!/' \
  -e '/^mozilla\/emSign_ECC_Root_CA_-_G3\.crt$/s/^/!/' \
  /etc/ca-certificates.conf
update-ca-certificates --fresh

export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
# タスクイメージに古いBunが含まれる場合も、指定した版へ更新する。
curl -fsSL https://bun.sh/install | bash -s -- bun-v1.4.4
export PATH="$BUN_INSTALL/bin:$PATH"
ln -sf "$BUN_INSTALL/bin/bun" /usr/local/bin/bun

rm -rf "$APP_DIR"

if [ -n "${DEEPAGENTS_TBENCH_BUNDLE_BASE64:-}" ]; then
  mkdir -p "$APP_DIR"
  printf '%s' "$DEEPAGENTS_TBENCH_BUNDLE_BASE64" | base64 -d > /tmp/deepagents-tbench-autotune-ts.tgz
  tar -xzf /tmp/deepagents-tbench-autotune-ts.tgz -C "$APP_DIR"
elif [ -n "${DEEPAGENTS_TBENCH_REPO:-}" ]; then
  git clone "$DEEPAGENTS_TBENCH_REPO" "$APP_DIR"
  cd "$APP_DIR"
  git checkout "${DEEPAGENTS_TBENCH_REF:-main}"
else
  echo "DEEPAGENTS_TBENCH_REPO or embedded bundle is required" >&2
  exit 1
fi

cd "$APP_DIR"

if [ -f bun.lock ]; then
  bun install --frozen-lockfile
else
  bun install
fi
