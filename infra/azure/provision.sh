#!/usr/bin/env bash
# provision.sh — stand up (or re-stand-up, for failover) the Brachiomimus
# VLA policy-server VM on Azure.
#
# Usage:
#   TAILSCALE_AUTHKEY=tskey-... ./provision.sh --location westus
#   TAILSCALE_AUTHKEY=tskey-... ./provision.sh --location westus2   # standby/failover region
#   ./provision.sh --location westus --dry-run                     # print az commands, do nothing
#
# See docs/vla-inference-azure.md for the full walkthrough. This script only
# provisions the VM (Docker + NVIDIA container toolkit + Tailscale, via
# cloud-init-policy-server.yaml); starting the PolicyServer container is a
# separate step documented there.
#
# Requires: az CLI (logged in, correct subscription selected), envsubst
# (part of gettext).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOCATION=""
RESOURCE_GROUP="brachiomimus-inference"
VM_NAME="brachiomimus-policy-server"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --location) LOCATION="$2"; shift 2 ;;
    --resource-group) RESOURCE_GROUP="$2"; shift 2 ;;
    --vm-name) VM_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$LOCATION" ]]; then
  echo "Usage: $0 --location <westus|westus2> [--resource-group NAME] [--vm-name NAME] [--dry-run]" >&2
  exit 1
fi

if [[ "$DRY_RUN" != true && -z "${TAILSCALE_AUTHKEY:-}" ]]; then
  echo "Set TAILSCALE_AUTHKEY (a reusable, tagged auth key from the Tailscale admin console)" >&2
  echo "before running for real. Never commit this key — pass it as an env var only." >&2
  exit 1
fi

run() {
  if [[ "$DRY_RUN" == true ]]; then
    printf '[dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

# --- 1. Quota check --------------------------------------------------------
# Azure tracks vCPU quota (the "NCasT4v3 Family" line) and raw GPU quota
# separately — a subscription can show plenty of the former and still be at
# 0 GPUs, which fails VM creation with a confusing error. There's no single
# stable field name to grep for across API versions, so this surfaces the
# raw usage table and asks you to eyeball both lines rather than risk a
# silent false-pass from a brittle string match.
echo "== Quota in $LOCATION (confirm a T4-family vCPU line >= 4 AND a separate GPUs line >= 1) =="
az vm list-usage --location "$LOCATION" -o table | grep -iE "t4|gpu|Name" || true
echo
if [[ "$DRY_RUN" != true ]]; then
  read -r -p "Both lines show enough headroom? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting. Request the missing quota first (see docs/vla-inference-azure.md)." >&2
    exit 1
  fi
fi

# --- 2. Resource group ------------------------------------------------------
run az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# --- 3. Render cloud-init with the real Tailscale auth key -----------------
# Only substitute TAILSCALE_AUTHKEY — the file also contains legitimate
# shell substitutions ($(dpkg --print-architecture) etc.) meant to run
# *inside* the VM at boot, which must survive untouched.
RENDERED_CLOUD_INIT="$(mktemp)"
trap 'rm -f "$RENDERED_CLOUD_INIT"' EXIT
if [[ "$DRY_RUN" == true ]]; then
  TAILSCALE_AUTHKEY="<dry-run-placeholder>"
fi
TAILSCALE_AUTHKEY="$TAILSCALE_AUTHKEY" envsubst '${TAILSCALE_AUTHKEY}' \
  < "$SCRIPT_DIR/cloud-init-policy-server.yaml" > "$RENDERED_CLOUD_INIT"

# --- 4. Create the VM --------------------------------------------------------
# --nsg-rule NONE: no inbound rule is added (not even SSH) — the VM is
# reachable only over the Tailscale mesh once cloud-init joins it. A public
# IP is still attached (needed for outbound apt/Docker Hub/Tailscale/HF Hub
# traffic during setup) but nothing can reach it, since Azure NSGs deny all
# inbound by default absent an explicit allow rule.
run az vm create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --location "$LOCATION" \
  --image Ubuntu2204 \
  --size Standard_NC4as_T4_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --nsg-rule NONE \
  --custom-data "$RENDERED_CLOUD_INIT"

# --- 5. NVIDIA GPU driver extension ------------------------------------------
# The container toolkit (installed by cloud-init) needs a real driver on the
# host underneath it.
run az vm extension set \
  --resource-group "$RESOURCE_GROUP" \
  --vm-name "$VM_NAME" \
  --name NvidiaGpuDriverLinux \
  --publisher Microsoft.HpcCompute

# --- 6. Auto-shutdown safety net --------------------------------------------
# Cost control, not a substitute for manually stopping the VM when done —
# see docs/vla-inference-azure.md#cost-control.
run az vm auto-shutdown \
  --resource-group "$RESOURCE_GROUP" \
  --name "$VM_NAME" \
  --time 0300

echo
echo "Done. Check the Tailscale admin console for '$VM_NAME' joining the tailnet,"
echo "then follow docs/vla-inference-azure.md to start the PolicyServer container."
