"""Draws the JeuVoice app icon: a Frost-dark tile, white "JEU", orange soundwave.

Run it with the engine Python, which has Pillow:

    engine\\.venv\\Scripts\\python packaging\\windows\\make-icon.py

It writes packaging\\windows\\icon.png (1024 px) and icon.ico (16-256 px).
The exe embeds icon.ico through build.rs; cargo-packager uses it for the
installer. Keep the drawing flat and bold: it must read at 16 px.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 1024
HERE = Path(__file__).resolve().parent
FONT = r"C:\Windows\Fonts\ariblk.ttf"  # Arial Black
TEXT = "JEU"
WHITE = (255, 255, 255, 255)
ORANGE = (255, 159, 90, 255)       # style.rs accent #ff9f5a
ORANGE_LIGHT = (255, 184, 132, 255)
TILE_TOP = (27, 28, 34)
TILE_BOTTOM = (15, 16, 20)


def tile() -> Image.Image:
    """The rounded tile with a soft vertical gradient."""
    gradient = Image.new("RGBA", (S, S))
    draw = ImageDraw.Draw(gradient)
    for y in range(S):
        t = y / (S - 1)
        color = tuple(round(TILE_TOP[i] + (TILE_BOTTOM[i] - TILE_TOP[i]) * t) for i in range(3))
        draw.line([(0, y), (S, y)], fill=(*color, 255))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=210, fill=255)
    image = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    image.paste(gradient, (0, 0), mask)
    # A thin light edge, like the glass panels in the app.
    ImageDraw.Draw(image).rounded_rectangle(
        [2, 2, S - 3, S - 3], radius=208, outline=(255, 255, 255, 18), width=3
    )
    return image


def main() -> None:
    image = tile()
    draw = ImageDraw.Draw(image)

    # "JEU": Arial Black, tightly spaced, centered in the top half.
    font = ImageFont.truetype(FONT, 250)
    spacing = 4
    widths = [draw.textlength(ch, font=font) for ch in TEXT]
    x = (S - (sum(widths) + spacing * (len(TEXT) - 1))) / 2
    for ch, width in zip(TEXT, widths):
        draw.text((x, 150), ch, font=font, fill=WHITE)
        x += width + spacing

    # The soundwave: five rounded bars, the middle one a little brighter.
    heights = [140, 260, 350, 260, 140]
    bar_w, gap, bottom = 92, 52, 880
    x = (S - (len(heights) * bar_w + (len(heights) - 1) * gap)) / 2
    for i, height in enumerate(heights):
        color = ORANGE_LIGHT if i == 2 else ORANGE
        draw.rounded_rectangle(
            [x, bottom - height, x + bar_w, bottom], radius=bar_w / 2, fill=color
        )
        x += bar_w + gap

    image.save(HERE / "icon.png")
    image.save(
        HERE / "icon.ico",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {HERE / 'icon.png'} and {HERE / 'icon.ico'}")


if __name__ == "__main__":
    main()
