#!/usr/bin/env bash
# Build the Mac App Store package of كلمة:
#
#   $DIST/kalima-<version>-mac-app-store.pkg     signed installer package for App Store Connect
#
# It differs from the website build (package.sh) where the App Store requires it:
# - sandboxed (mas/entitlements.plist), with the App Store bundle ID and provisioning profile;
# - no kalima-cli: a sandboxed app can't write other AI apps' settings, so the MCP server and the
#   one-button AI connection stay in the website build;
# - no self-updater: the App Store updates the app (craft_update::store sees the sandbox).
#
# Usage: packaging/macos/package-mas.sh [--skip-build]
#
# Env:
#   MAS_BUNDLE_ID          App Store bundle ID (default com.mokteb.kalima)
#   MAS_PROVISION_PROFILE  path of the Mac App Store .provisionprofile for that bundle ID
#   MAS_APP_IDENTITY       codesign identity for the app ("Apple Distribution" or
#                          "3rd Party Mac Developer Application"); import-mas-cert.sh sets it
#   MAS_INSTALLER_IDENTITY productbuild identity ("3rd Party Mac Developer Installer")
#   MACOS_KEYCHAIN         keychain holding both identities (optional)
# Upload (optional; all three, App Store Connect API key with the App Manager role):
#   ASC_KEY_ID, ASC_ISSUER_ID, ASC_KEY_PATH (the .p8 file)
set -euo pipefail
# shellcheck source=../env.sh
. "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
HERE="$ROOT/packaging/macos"

SKIP_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1; shift ;;
    -h | --help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

BUNDLE_ID="${MAS_BUNDLE_ID:-com.mokteb.kalima}"
for v in MAS_PROVISION_PROFILE MAS_APP_IDENTITY MAS_INSTALLER_IDENTITY; do
  if [[ -z "${!v:-}" ]]; then
    echo "error: $v is not set (see the header of $0)" >&2
    exit 1
  fi
done
if [[ ! -s "$MAS_PROVISION_PROFILE" ]]; then
  echo "error: no provisioning profile at $MAS_PROVISION_PROFILE" >&2
  exit 1
fi

# Keep in sync with LSMinimumSystemVersion in Info.plist.in (package.sh uses the same).
export MACOSX_DEPLOYMENT_TARGET=11.0
SHORT_VERSION="${VERSION%%-*}"
TARGETS=(aarch64-apple-darwin x86_64-apple-darwin)
WORK="$CARGO_TARGET_DIR/mas-package"
APP="$WORK/كلمة.app"
PKG="$DIST/kalima-$VERSION-mac-app-store.pkg"
kc=()
[[ -n "${MACOS_KEYCHAIN:-}" ]] && kc=(--keychain "$MACOS_KEYCHAIN")

echo "==> كلمة $VERSION for the Mac App Store ($BUNDLE_ID)"

if [[ "$SKIP_BUILD" = 0 ]]; then
  args=()
  for t in "${TARGETS[@]}"; do args+=(--target "$t"); done
  (cd "$ROOT" && cargo build --release --locked -p wordcraft "${args[@]}")
fi

rm -rf "$WORK"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
inputs=()
for t in "${TARGETS[@]}"; do inputs+=("$CARGO_TARGET_DIR/$t/release/wordcraft"); done
lipo -create -output "$APP/Contents/MacOS/WordCraft" "${inputs[@]}"
lipo -info "$APP/Contents/MacOS/WordCraft"
cp "$ROOT/assets/app-icon/wordcraft.icns" "$APP/Contents/Resources/WordCraft.icns"
cp "$ROOT/LICENSE-MIT" "$ROOT/LICENSE-APACHE" "$ROOT/NOTICE" "$ROOT/ATTRIBUTION.md" "$ROOT"/assets/fonts/OFL-*.txt "$APP/Contents/Resources/"
copy_docs "$APP/Contents/Resources"
cp "$MAS_PROVISION_PROFILE" "$APP/Contents/embedded.provisionprofile"

PLIST="$APP/Contents/Info.plist"
sed -e "s/@VERSION@/$VERSION/g" -e "s/@SHORT_VERSION@/$SHORT_VERSION/g" \
  -e "s/@BUILD_SHA@/${WORDCRAFT_BUILD_SHA:-unknown}/g" \
  "$HERE/Info.plist.in" >"$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier $BUNDLE_ID" "$PLIST"
# Only the system's HTTPS is used, which is exempt from export compliance paperwork.
/usr/libexec/PlistBuddy -c "Add :ITSAppUsesNonExemptEncryption bool false" "$PLIST"
plutil -lint "$PLIST"
printf 'APPL????' >"$APP/Contents/PkgInfo"

# The profile must be for this bundle ID, or App Store Connect rejects the upload.
profile_app_id="$(security cms -D -i "$MAS_PROVISION_PROFILE" | plutil -extract Entitlements.com\\.apple\\.application-identifier raw -o - -)"
case "$profile_app_id" in
  *."$BUNDLE_ID") ;;
  *) echo "error: the provisioning profile is for $profile_app_id, not $BUNDLE_ID" >&2; exit 1 ;;
esac
team_id="${profile_app_id%%.*}"

# The sandbox entitlements plus the application and team identifiers the profile grants.
ENT="$WORK/entitlements.plist"
cp "$HERE/mas/entitlements.plist" "$ENT"
/usr/libexec/PlistBuddy -c "Add :com.apple.application-identifier string $profile_app_id" "$ENT"
/usr/libexec/PlistBuddy -c "Add :com.apple.developer.team-identifier string $team_id" "$ENT"

xattr -cr "$APP"
codesign --force --timestamp --options runtime --sign "$MAS_APP_IDENTITY" ${kc[@]+"${kc[@]}"} \
  --entitlements "$ENT" "$APP"
codesign --verify --strict --deep --verbose=2 "$APP"
codesign -d --entitlements - "$APP"

echo "==> building $PKG"
rm -f "$PKG"
productbuild --component "$APP" /Applications --sign "$MAS_INSTALLER_IDENTITY" ${kc[@]+"${kc[@]}"} "$PKG"
pkgutil --check-signature "$PKG"
ls -lh "$PKG"

if [[ -n "${ASC_KEY_ID:-}" ]] && [[ -n "${ASC_ISSUER_ID:-}" ]] && [[ -s "${ASC_KEY_PATH:-}" ]]; then
  # altool looks for the key as AuthKey_<id>.p8 in ~/.appstoreconnect/private_keys.
  keys="$HOME/.appstoreconnect/private_keys"
  mkdir -p "$keys"
  cp "$ASC_KEY_PATH" "$keys/AuthKey_$ASC_KEY_ID.p8"
  echo "==> validating and uploading to App Store Connect"
  xcrun altool --validate-app -f "$PKG" -t macos --apiKey "$ASC_KEY_ID" --apiIssuer "$ASC_ISSUER_ID"
  xcrun altool --upload-app -f "$PKG" -t macos --apiKey "$ASC_KEY_ID" --apiIssuer "$ASC_ISSUER_ID"
  rm -f "$keys/AuthKey_$ASC_KEY_ID.p8"
else
  warn "ASC_KEY_ID / ASC_ISSUER_ID / ASC_KEY_PATH not set: the package was built but not uploaded"
fi
echo "==> done"
