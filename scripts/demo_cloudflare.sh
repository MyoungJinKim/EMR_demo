#!/bin/bash
# Keep this demo's Cloudflare login and logs inside its private tools directory.
set -euo pipefail
umask 077
DEMO_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
export XDG_CONFIG_HOME="$DEMO_PROJECT/.demo-tools/cloudflare-config"
export WRANGLER_SEND_METRICS=false
export CLOUDFLARED_PATH="$DEMO_PROJECT/.demo-tools/cloudflared"
mkdir -p "$XDG_CONFIG_HOME"
chmod 700 "$XDG_CONFIG_HOME"
exec node "$DEMO_PROJECT/.demo-tools/cloudflare-cli/node_modules/wrangler/bin/wrangler.js" "$@"
