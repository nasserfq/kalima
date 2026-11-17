# كلمة in the Mac App Store

`../package-mas.sh` turns a release build into `kalima-<version>-mac-app-store.pkg`, a sandboxed,
App Store signed installer for App Store Connect. The release workflow builds it on every push to
`release` when its secrets exist, keeps it as the **mac-app-store** workflow artifact, and uploads it
to App Store Connect when the API key secrets exist too. The website DMG is unchanged.

## What changes in the App Store copy

- It runs in the App Sandbox (`entitlements.plist` here): it opens and saves the documents people
  pick, drop or open from Finder, and can use the internet (the AI assistant panel) and printers.
- Its bundle ID is `com.mokteb.kalima` (`MAS_BUNDLE_ID`), separate from the website build.
- It never checks for or downloads updates; Help › Check for Updates opens the App Store's updates
  page (`craft_update::store` sees the sandbox).
- It has no `kalima-cli`: a sandboxed app can't edit other AI apps' settings, so the MCP server and
  «ربط مع الذكاء الاصطناعي» stay in the website build. The window says so.
- Known gap: the sandbox forgets permission to a file after the app quits, so reopening a file from
  the recent list after a restart needs the Open panel (security-scoped bookmarks are not in yet).

## Secrets (Settings › Secrets and variables › Actions)

| Secret | What |
|---|---|
| `MAS_CERTIFICATE` | base64 of one `.p12` with the **Mac App Distribution** and **Mac Installer Distribution** certificates and their keys |
| `MAS_CERTIFICATE_PASSWORD` | the `.p12` password |
| `MAS_PROVISION_PROFILE` | base64 of the **Mac App Store Connect** provisioning profile for `com.mokteb.kalima` |
| `ASC_KEY_ID`, `ASC_ISSUER_ID` | the App Store Connect API key (App Manager role) |
| `ASC_KEY` | the text of its `AuthKey_<id>.p8` file |

On a Mac: `base64 -i file.p12 | pbcopy` copies a file as base64.

Without the first three, no package is built; without the last three, it is built but not
uploaded. Either way the website downloads are not affected.
