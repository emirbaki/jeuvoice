"""Draws the JeuVoice app icon: black tile, white "JEU", orange soundwave.

The "JEU" letters are hand-drawn SVG paths with an orange 3D layer stack.
The soundwave is a row of rounded orange bars. Run it with a Python that has
Pillow and resvg-py (pip install resvg-py):

    engine\\.venv\\Scripts\\python packaging\\windows\\make-icon.py

It writes packaging\\windows\\icon.svg (the source), icon.png (1024 px) and
icon.ico (16-256 px). The exe embeds icon.ico through build.rs; cargo-packager
uses it for the installer.
"""

import io
from pathlib import Path

import resvg_py
from PIL import Image

HERE = Path(__file__).resolve().parent

# Letter shapes on a 1024 canvas. The even-odd fill cuts the E and U slots.
J = ("M160 150H300V450A110 160 0 0 1 190 610A110 160 0 0 1 80 450"
     "V440H140A20 20 0 0 0 160 420Z")
E = ("M450 150H580V262H495A30 30 0 0 0 495 322H580V438H495A30 30 0 0 0 495 498H580V610"
     "H450A120 120 0 0 1 330 490V270A120 120 0 0 1 450 150Z")
U = ("M610 150H750V420A27 27 0 0 0 804 420V150H944V440"
     "A167 170 0 0 1 610 440Z")

# The 3D stack under the letters: offset (px down) and color, back to front.
LAYERS = [(88, "#7A2E06"), (66, "#B8470B"), (44, "#E8650F"), (22, "#FF9A2E")]

# Soundwave bar heights, left half and the middle bar. The right half mirrors it.
WAVE_HALF = [16, 30, 52, 34, 70, 104, 76, 130, 160, 118, 180]
WAVE_STEP, WAVE_Y = 24, 845


def svg() -> str:
    letters = (f'<path transform="translate(-26 0)" d="{J}"/><path d="{E}"/>'
               f'<path transform="translate(26 0)" d="{U}"/>')
    stack = "".join(
        f'<g transform="translate(0 {dy})" fill="{c}" stroke="{c}">{letters}</g>'
        for dy, c in LAYERS)

    heights = WAVE_HALF + WAVE_HALF[-2::-1]
    x0 = 512 - WAVE_STEP * (len(heights) // 2)
    bars = "".join(
        f'<line x1="{x0 + i * WAVE_STEP}" y1="{WAVE_Y - v / 2:.0f}" '
        f'x2="{x0 + i * WAVE_STEP}" y2="{WAVE_Y + v / 2:.0f}"/>'
        for i, v in enumerate(heights))

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
<defs>
  <radialGradient id="bg" cx="50%" cy="42%" r="70%">
    <stop offset="0" stop-color="#1A1A1A"/><stop offset="1" stop-color="#000"/>
  </radialGradient>
  <radialGradient id="glow" cx="50%" cy="50%" r="50%">
    <stop offset="0" stop-color="#FF7A1A" stop-opacity=".22"/><stop offset="1" stop-color="#FF7A1A" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="wave" x1="0" y1="760" x2="0" y2="930" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="#FFC15E"/><stop offset="1" stop-color="#FF5A00"/>
  </linearGradient>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="160%">
    <feDropShadow dx="0" dy="18" stdDeviation="22" flood-color="#000" flood-opacity=".8"/>
  </filter>
</defs>
<rect width="1024" height="1024" fill="url(#bg)"/>
<ellipse cx="512" cy="{WAVE_Y}" rx="300" ry="120" fill="url(#glow)"/>
<g transform="translate(0 -8) translate(512 380) scale(0.9) translate(-512 -380)" fill-rule="evenodd" stroke-linejoin="round" stroke-width="16">
  <g filter="url(#shadow)">{stack}</g>
  <g fill="#FFFFFF" stroke="#FFFFFF">{letters}</g>
</g>
<g stroke="url(#wave)" stroke-width="14" stroke-linecap="round">{bars}</g>
</svg>
'''


def main() -> None:
    source = svg()
    (HERE / "icon.svg").write_text(source, encoding="utf-8")
    image = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=source))))
    image.save(HERE / "icon.png")
    image.save(
        HERE / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {HERE / 'icon.svg'}, {HERE / 'icon.png'} and {HERE / 'icon.ico'}")


if __name__ == "__main__":
    main()
