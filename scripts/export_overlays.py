"""
Export Overlays Script
Generate transparent PNG overlays for video editing
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_font(size: int):
    """Get font with fallback"""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    return ImageFont.load_default()


def create_scoreboard_overlay(width: int = 300, height: int = 150) -> Image.Image:
    """Create transparent scoreboard overlay"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent background
    draw.rounded_rectangle([10, 10, width - 10, height - 10],
                           radius=15, fill=(30, 40, 60, 200))

    # Border
    draw.rounded_rectangle([10, 10, width - 10, height - 10],
                           radius=15, outline=(100, 150, 200, 255), width=2)

    # Labels
    font_large = get_font(32)
    font_small = get_font(16)

    draw.text((width // 2, 35), "SCORE", font=font_small,
              fill=(150, 180, 220, 255), anchor="mt")
    draw.text((width // 2, 65), "0000", font=font_large,
              fill=(255, 255, 255, 255), anchor="mt")

    draw.text((50, 105), "Step: 000", font=font_small,
              fill=(180, 180, 180, 255), anchor="lt")
    draw.text((width - 50, 105), "ε: 0.00", font=font_small,
              fill=(180, 180, 180, 255), anchor="rt")

    return img


def create_pieces_overlay(width: int = 400, height: int = 120) -> Image.Image:
    """Create transparent pieces panel overlay"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent background
    draw.rounded_rectangle([5, 5, width - 5, height - 5],
                           radius=10, fill=(30, 40, 60, 180))

    # Three piece slots
    slot_width = (width - 40) // 3
    for i in range(3):
        x = 15 + i * (slot_width + 5)
        draw.rounded_rectangle([x, 15, x + slot_width, height - 15],
                               radius=8, fill=(40, 55, 80, 200),
                               outline=(80, 100, 140, 255))

    return img


def create_qvalues_overlay(width: int = 250, height: int = 200) -> Image.Image:
    """Create transparent Q-values overlay"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent background
    draw.rounded_rectangle([5, 5, width - 5, height - 5],
                           radius=10, fill=(30, 40, 60, 200))

    font_medium = get_font(18)
    font_small = get_font(14)

    # Title
    draw.text((width // 2, 20), "TOP Q-VALUES", font=font_medium,
              fill=(255, 220, 80, 255), anchor="mt")

    # Sample entries
    y = 50
    for i in range(4):
        draw.text((20, y), f"#{i+1}", font=font_small,
                  fill=(150, 150, 150, 255), anchor="lt")
        draw.text((50, y), f"p{i}@(x,y)", font=font_small,
                  fill=(200, 200, 200, 255), anchor="lt")
        draw.text((width - 20, y), f"0.00", font=font_small,
                  fill=(100, 255, 100, 255), anchor="rt")
        y += 30

    return img


def create_graph_overlay(width: int = 400, height: int = 250) -> Image.Image:
    """Create transparent graph area overlay"""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Semi-transparent background
    draw.rounded_rectangle([5, 5, width - 5, height - 5],
                           radius=10, fill=(20, 30, 50, 200))

    # Border
    draw.rounded_rectangle([5, 5, width - 5, height - 5],
                           radius=10, outline=(60, 80, 120, 255), width=2)

    font_medium = get_font(18)

    # Title
    draw.text((width // 2, 20), "TRAINING PROGRESS", font=font_medium,
              fill=(255, 220, 80, 255), anchor="mt")

    # Graph area placeholder
    draw.rectangle([30, 50, width - 30, height - 40],
                   outline=(80, 100, 140, 200))

    # Axis labels
    font_tiny = get_font(12)
    draw.text((width // 2, height - 20), "Episode", font=font_tiny,
              fill=(150, 150, 150, 255), anchor="mt")

    return img


def main():
    parser = argparse.ArgumentParser(description='Export overlay PNGs')
    parser.add_argument('--out', type=str, default='outputs/overlays',
                        help='Output directory')

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BLOCK BLAST - EXPORT OVERLAYS")
    print(f"{'='*60}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}\n")

    # Generate overlays
    overlays = {
        'scoreboard_overlay.png': create_scoreboard_overlay(),
        'pieces_overlay.png': create_pieces_overlay(),
        'qvalues_overlay.png': create_qvalues_overlay(),
        'graph_overlay.png': create_graph_overlay(),
    }

    for name, img in overlays.items():
        path = out_dir / name
        img.save(path)
        print(f"Created: {path}")

    print(f"\nGenerated {len(overlays)} overlays")
    print("Done!")


if __name__ == "__main__":
    main()
