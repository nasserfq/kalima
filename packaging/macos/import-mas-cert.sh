#!/usr/bin/env bash
# CI only: import the Mac App Store certificates (app + installer, one .p12) and the provisioning
# profile, for package-mas.sh.
#
# Inputs (env): MAS_CERTIFICATE (base64 .p12 holding the Mac App Distribution and Mac Installer
# Distribution certificates with their keys), MAS_CERTIFICATE_PASSWORD, MAS_PROVISION_PROFILE_B64
# (base64 .provisionprofile), ASC_KEY (contents of the App Store Connect .p8, optional),
# RUNNER_TEMP (or TMPDIR).
# Outputs (appended to $GITHUB_ENV in Actions, otherwise printed): MAS_APP_IDENTITY,
# MAS_INSTALLER_IDENTITY, MACOS_KEYCHAIN, MAS_PROVISION_PROFILE, and ASC_KEY_PATH when ASC_KEY is set.
#
# Missing MAS_CERTIFICATE: prints a warning and exits 0; the workflow then skips the package.
set -euo pipefail
umask 077
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

if [[ -z "${MAS_CERTIFICATE:-}" ]] || [[ -z "${MAS_PROVISION_PROFILE_B64:-}" ]]; then
  warn "MAS_CERTIFICATE or MAS_PROVISION_PROFILE_B64 is not set: no Mac App Store package"
  exit 0
fi

TMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
KC_PASS="$(openssl rand -hex 24)"
# A fresh private directory with an unpredictable name, as in import-cert.sh (#3692).
WORK="$(mktemp -d "$TMP/kalima-mas.XXXXXXXX")"
CERT="$WORK/mas.p12"
KEYCHAIN="$WORK/mas.keychain-db"
PROFILE="$WORK/mas.provisionprofile"
cleanup() {
  local status=$?
  rm -f "$CERT"
  if [[ "$status" -ne 0 ]]; then
    security delete-keychain "$KEYCHAIN" 2>/dev/null || true
    rm -rf "$WORK"
  fi
  exit "$status"
}
trap cleanup EXIT
printf '%s' "$MAS_CERTIFICATE" | base64 --decode >"$CERT"
printf '%s' "$MAS_PROVISION_PROFILE_B64" | base64 --decode >"$PROFILE"
security create-keychain -p "$KC_PASS" "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p "$KC_PASS" "$KEYCHAIN"
security import "$CERT" -P "${MAS_CERTIFICATE_PASSWORD:-}" -T /usr/bin/codesign -T /usr/bin/productbuild \
  -T /usr/bin/security -f pkcs12 -k "$KEYCHAIN"
rm -f "$CERT"
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KC_PASS" "$KEYCHAIN" >/dev/null
# shellcheck disable=SC2046
security list-keychains -d user -s "$KEYCHAIN" $(security list-keychains -d user | tr -d '"')

# Identities by certificate name: Apple has used both names for each kind.
APP_ID="$(security find-identity -v -p codesigning "$KEYCHAIN" |
  awk '/"(Apple Distribution|3rd Party Mac Developer Application):/ { print $2; exit }')"
INSTALLER_ID="$(security find-identity -v "$KEYCHAIN" |
  awk '/"(3rd Party Mac Developer Installer|Mac Installer Distribution):/ { print $2; exit }')"
if [[ -z "$APP_ID" ]] || [[ -z "$INSTALLER_ID" ]]; then
  security find-identity -v "$KEYCHAIN" >&2
  echo "error: MAS_CERTIFICATE needs both a Mac App Distribution and a Mac Installer Distribution certificate" >&2
  exit 1
fi
echo "Imported app identity $APP_ID and installer identity $INSTALLER_ID"

out=("MAS_APP_IDENTITY=$APP_ID" "MAS_INSTALLER_IDENTITY=$INSTALLER_ID" "MACOS_KEYCHAIN=$KEYCHAIN"
  "MAS_PROVISION_PROFILE=$PROFILE")
if [[ -n "${ASC_KEY:-}" ]]; then
  printf '%s\n' "$ASC_KEY" >"$WORK/asc.p8"
  out+=("ASC_KEY_PATH=$WORK/asc.p8")
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  printf '%s\n' "${out[@]}" >>"$GITHUB_ENV"
else
  printf '%s\n' "${out[@]}"
fi
