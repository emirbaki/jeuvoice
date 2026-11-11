#!/usr/bin/env bash
# Makes the macOS app bundle and disk image on an Apple Silicon Mac.
#
#   scripts/make-dmg.sh
#
# Output: dist/macos/JeuVoice.app and dist/macos/JeuVoice_<version>_aarch64.dmg.
# The script downloads the pinned Python and uv once and checks the SHA-256.
# It never touches ~/Documents/JeuVoice or the app data folder.
# Signing and notarization: see packaging/macos/README.md.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Run this script on an Apple Silicon Mac." >&2
    exit 1
fi

# Pinned downloads. A new version means a new tag, file and SHA-256.
PY_TAG="20261110"
PY_ASSET="cpython-3.12.16+${PY_TAG}-aarch64-apple-darwin-install_only.tar.gz"
PY_SHA="e956e60bc9465d577ef8062059aa05cd3a2665b4b8678a2ed24e432368330b8f"
PY_NAME="python-3.12.16-${PY_TAG}.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_TAG}/${PY_ASSET//+/%2B}"

UV_VERSION="0.13.7"
UV_ASSET="uv-aarch64-apple-darwin.tar.gz"
UV_SHA="81b811b57e4624df7ed3aac148e0359100d9962f3d9dce00aec533dc2e086731"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_ASSET}"

OUT="$REPO/dist/macos"
RES="$OUT/resources"
CACHE="$REPO/dist/download-cache"
mkdir -p "$CACHE"

# Download $1 to $3 once, and check that its SHA-256 is $2.
pinned() {
    local url="$1" sha="$2" dest="$3"
    if [[ -f "$dest" ]] && [[ "$(shasum -a 256 "$dest" | cut -d' ' -f1)" == "$sha" ]]; then
        echo "using $dest"
        return
    fi
    echo "downloading $url"
    curl -fL --retry 3 -o "$dest" "$url"
    local got
    got="$(shasum -a 256 "$dest" | cut -d' ' -f1)"
    if [[ "$got" != "$sha" ]]; then
        rm -f "$dest"
        echo "SHA-256 mismatch for $url: expected $sha, got $got" >&2
        exit 1
    fi
}

echo "1/4 Build the release app (whisper uses Metal on macOS)."
MACOSX_DEPLOYMENT_TARGET=13.0 cargo build --release -p jeuvoice-app

echo "2/4 Collect the resources in $RES."
rm -rf "$OUT"
mkdir -p "$RES"
pinned "$PY_URL" "$PY_SHA" "$CACHE/$PY_ASSET"
cp "$CACHE/$PY_ASSET" "$RES/$PY_NAME"
pinned "$UV_URL" "$UV_SHA" "$CACHE/$UV_ASSET"
tar -xzf "$CACHE/$UV_ASSET" -C "$CACHE"
cp "$CACHE/uv-aarch64-apple-darwin/uv" "$RES/uv"
chmod +x "$RES/uv"
rsync -a --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    --exclude .mypy_cache --exclude tests "$REPO/engine/" "$RES/engine/"
cp "$REPO/engine/requirements-macos.txt" "$RES/"
cp -R "$REPO/assets" "$RES/assets"

for rel in uv "$PY_NAME" requirements-macos.txt engine/jeuvoice_engine/__main__.py \
    engine/overrides.txt assets/presets assets/recipes; do
    [[ -e "$RES/$rel" ]] || { echo "the bundle is incomplete: $rel is missing" >&2; exit 1; }
done
[[ ! -e "$RES/engine/.venv" ]] || { echo "the dev venv leaked into the bundle" >&2; exit 1; }

echo "3/4 Make the packager config."
VERSION="$(cargo metadata --no-deps --format-version 1 \
    | python3 -c 'import json,sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"]=="jeuvoice-app"))')"
cat > "$OUT/packager.json" <<EOF
{
  "name": "jeuvoice-app",
  "productName": "JeuVoice",
  "version": "$VERSION",
  "identifier": "com.jeu.jeuvoice",
  "description": "Speech, transcription and voice training in one window.",
  "outDir": "$OUT",
  "binariesDir": "$REPO/target/release",
  "binaries": [{ "path": "jeuvoice-app", "main": true }],
  "formats": ["app", "dmg"],
  "resources": [
    "$RES/$PY_NAME",
    "$RES/uv",
    "$RES/requirements-macos.txt",
    "$RES/engine",
    "$RES/assets"
  ],
  "macos": {
    "minimumSystemVersion": "13.0",
    "infoPlistPath": "$REPO/packaging/macos/Info.plist"
  }
}
EOF

echo "4/4 Make JeuVoice.app and the dmg."
cargo packager --config "$OUT/packager.json"

echo
echo "Done:"
ls -1 "$OUT" | grep -E '\.(app|dmg)$' | sed "s|^|  $OUT/|"
echo "Next: sign and notarize (packaging/macos/README.md), then test on a clean user account."
