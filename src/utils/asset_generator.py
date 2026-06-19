"""
============================================
SignBridge - Placeholder Sign Asset Generator
============================================
Creates simple placeholder images for each
sign (A-Z, 0-9, common words). Each image
shows the sign letter/word in a styled card.

These are PLACEHOLDERS - you should replace
them with actual ASL reference images later.

HOW TO USE:
  python -m src.utils.asset_generator
============================================
"""

import os
from PIL import Image, ImageDraw, ImageFont

from src.utils.config import ALL_LABELS, SIGNS_DIR


def generate_sign_images():
    """
    Generate a placeholder sign image for each label.
    Creates a clean card-style image with the sign text.
    """
    print("\n" + "=" * 50)
    print("  SignBridge - Sign Image Generator")
    print("=" * 50)

    os.makedirs(SIGNS_DIR, exist_ok=True)

    # Image dimensions
    IMG_WIDTH = 400
    IMG_HEIGHT = 400

    # Color scheme (dark blue gradient look)
    BG_COLOR = (30, 36, 50)           # Dark navy background
    CARD_COLOR = (45, 55, 75)         # Slightly lighter card
    ACCENT_COLOR = (0, 200, 180)      # Teal accent
    TEXT_COLOR = (255, 255, 255)       # White text
    SUBTITLE_COLOR = (150, 170, 200)  # Light blue-gray

    for label in ALL_LABELS:
        # Create image
        img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        # Draw rounded card background
        card_margin = 20
        draw.rounded_rectangle(
            [card_margin, card_margin, IMG_WIDTH - card_margin, IMG_HEIGHT - card_margin],
            radius=20,
            fill=CARD_COLOR,
            outline=ACCENT_COLOR,
            width=2,
        )

        # Draw accent bar at top of card
        draw.rounded_rectangle(
            [card_margin, card_margin, IMG_WIDTH - card_margin, card_margin + 60],
            radius=20,
            fill=ACCENT_COLOR,
        )
        # Fix the bottom corners of the accent bar
        draw.rectangle(
            [card_margin, card_margin + 40, IMG_WIDTH - card_margin, card_margin + 60],
            fill=ACCENT_COLOR,
        )

        # Header text
        try:
            # Try to use a nice font (available on most systems)
            header_font = ImageFont.truetype("arial.ttf", 18)
            main_font = ImageFont.truetype("arial.ttf", 80)
            sub_font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            # Fallback to default font
            header_font = ImageFont.load_default()
            main_font = ImageFont.load_default()
            sub_font = ImageFont.load_default()

        # "ASL SIGN" header
        header_text = "ASL SIGN"
        header_bbox = draw.textbbox((0, 0), header_text, font=header_font)
        header_w = header_bbox[2] - header_bbox[0]
        draw.text(
            ((IMG_WIDTH - header_w) // 2, card_margin + 18),
            header_text,
            fill=BG_COLOR,
            font=header_font,
        )

        # Main sign letter/word (large, centered)
        display_text = label
        main_bbox = draw.textbbox((0, 0), display_text, font=main_font)
        main_w = main_bbox[2] - main_bbox[0]
        main_h = main_bbox[3] - main_bbox[1]
        main_x = (IMG_WIDTH - main_w) // 2
        main_y = (IMG_HEIGHT - main_h) // 2 - 10
        draw.text(
            (main_x, main_y),
            display_text,
            fill=TEXT_COLOR,
            font=main_font,
        )

        # Decorative hand emoji / subtitle
        subtitle = "✋ Hand Sign"
        sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        draw.text(
            ((IMG_WIDTH - sub_w) // 2, IMG_HEIGHT - card_margin - 50),
            subtitle,
            fill=SUBTITLE_COLOR,
            font=sub_font,
        )

        # Determine filename
        # For words with spaces, replace with underscore
        filename = label.replace(" ", "_") + ".png"
        filepath = os.path.join(SIGNS_DIR, filename)

        # Save image
        img.save(filepath)
        print(f"    Created: {filename}")

    print(f"\n  {len(ALL_LABELS)} sign images saved to: {SIGNS_DIR}")
    print("=" * 50 + "\n")


# ── Run when executed directly ──
if __name__ == "__main__":
    generate_sign_images()
