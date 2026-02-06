"""
Block Blast PIL Renderer for Video Export
Generates clean PNG frames with overlays for YouTube
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os

# Try to load a nice font, fallback to default
def get_font(size: int) -> ImageFont.FreeTypeFont:
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    return ImageFont.load_default()


# Colors
COLORS = {
    'bg': (45, 55, 85),
    'grid_bg': (28, 42, 70),
    'empty_cell': (35, 50, 85),
    'grid_line': (55, 75, 115),
    'text_white': (255, 255, 255),
    'text_yellow': (255, 220, 80),
    'text_green': (100, 255, 100),
    'text_red': (255, 100, 100),
    'highlight_clear': (255, 80, 80),
    'highlight_action': (255, 255, 100),
    'combo_bg': (30, 50, 90),
}

PIECE_COLORS = {
    'yellow': (255, 200, 50),
    'purple': (170, 120, 230),
    'green': (100, 220, 100),
    'orange': (255, 140, 50),
    'blue': (80, 160, 230),
    'red': (230, 80, 80),
    'cyan': (100, 220, 220),
    'pink': (230, 120, 170),
}


class BlockBlastRenderer:
    """PIL-based renderer for Block Blast frames"""

    def __init__(self, cell_size: int = 50, width: int = 800, height: int = 600):
        self.cell_size = cell_size
        self.width = width
        self.height = height
        self.grid_size = 8

        # Fonts
        self.font_large = get_font(36)
        self.font_medium = get_font(24)
        self.font_small = get_font(16)
        self.font_tiny = get_font(12)

        # Layout
        self.grid_padding = 30
        self.grid_x = self.grid_padding
        self.grid_y = 80
        self.grid_pixel_size = self.cell_size * self.grid_size

        # Panels
        self.info_x = self.grid_x + self.grid_pixel_size + 30
        self.pieces_y = self.grid_y + self.grid_pixel_size + 20

    def render_frame(self, step_info: Dict,
                     q_values_top: List[Dict] = None,
                     decision_tags: List[str] = None,
                     show_action: bool = True) -> Image.Image:
        """Render a single frame from step info"""

        img = Image.new('RGB', (self.width, self.height), COLORS['bg'])
        draw = ImageDraw.Draw(img)

        # Draw components
        self._draw_header(draw, step_info)
        self._draw_grid(draw, step_info, show_action)
        self._draw_pieces_panel(draw, step_info)
        self._draw_info_panel(draw, step_info, q_values_top, decision_tags)

        return img

    def _draw_header(self, draw: ImageDraw.Draw, info: Dict):
        """Draw score and step info at top"""
        # Score
        score = info.get('score_total', 0)
        draw.text((self.width // 2, 20), f"{score}",
                  font=self.font_large, fill=COLORS['text_white'], anchor="mt")

        # Score delta
        delta = info.get('score_delta', 0)
        if delta > 0:
            draw.text((self.width // 2 + 80, 25), f"+{delta}",
                      font=self.font_medium, fill=COLORS['text_green'], anchor="lt")

        # Step counter
        step = info.get('t', 0)
        draw.text((20, 20), f"Step {step}",
                  font=self.font_small, fill=COLORS['text_white'], anchor="lt")

        # Combo
        combo = info.get('combo_streak', 0)
        if combo > 0:
            draw.text((20, 45), f"Combo x{combo}",
                      font=self.font_medium, fill=COLORS['text_yellow'], anchor="lt")

    def _draw_grid(self, draw: ImageDraw.Draw, info: Dict, show_action: bool):
        """Draw the 8x8 game grid"""
        grid = info.get('grid', [[0]*8 for _ in range(8)])
        cleared_rows = info.get('cleared_rows', [])
        cleared_cols = info.get('cleared_cols', [])
        action = info.get('action', {})

        # Grid background
        draw.rounded_rectangle(
            [self.grid_x - 5, self.grid_y - 5,
             self.grid_x + self.grid_pixel_size + 5, self.grid_y + self.grid_pixel_size + 5],
            radius=10, fill=COLORS['grid_bg']
        )

        # Get action position if available
        action_cells = set()
        if show_action and action and action.get('piece_i') is not None:
            # We need piece shape - get from pieces info
            piece_idx = action.get('piece_i')
            ax, ay = action.get('x', -1), action.get('y', -1)
            pieces = info.get('pieces', [])
            if 0 <= piece_idx < len(pieces) and pieces[piece_idx]:
                from game.pieces import PIECES_DATA
                piece_id = pieces[piece_idx]
                if piece_id in PIECES_DATA:
                    for dx, dy in PIECES_DATA[piece_id]:
                        action_cells.add((ax + dx, ay + dy))

        # Draw cells
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cx = self.grid_x + x * self.cell_size
                cy = self.grid_y + y * self.cell_size
                cell_rect = [cx + 2, cy + 2, cx + self.cell_size - 2, cy + self.cell_size - 2]

                # Determine cell color
                if grid[y][x] == 1:
                    # Filled cell
                    color = (60, 60, 80)  # Default filled color

                    # Check if this was cleared
                    if y in cleared_rows or x in cleared_cols:
                        color = COLORS['highlight_clear']
                    # Check if this is the action
                    elif (x, y) in action_cells:
                        color = COLORS['highlight_action']

                    draw.rounded_rectangle(cell_rect, radius=4, fill=color)

                    # 3D effect for filled cells
                    highlight = tuple(min(255, c + 40) for c in color)
                    shadow = tuple(max(0, c - 40) for c in color)
                    draw.line([cx + 3, cy + 3, cx + self.cell_size - 4, cy + 3], fill=highlight, width=2)
                    draw.line([cx + 3, cy + 3, cx + 3, cy + self.cell_size - 4], fill=highlight, width=2)
                else:
                    # Empty cell
                    draw.rounded_rectangle(cell_rect, radius=4, fill=COLORS['empty_cell'])

        # Draw clear indicators
        for row in cleared_rows:
            y_pos = self.grid_y + row * self.cell_size + self.cell_size // 2
            draw.text((self.grid_x - 20, y_pos), f"R{row}",
                      font=self.font_tiny, fill=COLORS['text_red'], anchor="rm")

        for col in cleared_cols:
            x_pos = self.grid_x + col * self.cell_size + self.cell_size // 2
            draw.text((x_pos, self.grid_y + self.grid_pixel_size + 5), f"C{col}",
                      font=self.font_tiny, fill=COLORS['text_red'], anchor="mt")

    def _draw_pieces_panel(self, draw: ImageDraw.Draw, info: Dict):
        """Draw the 3 available pieces below the grid"""
        pieces = info.get('pieces', [None, None, None])
        piece_colors = info.get('piece_colors', [None, None, None])
        action = info.get('action', {})
        used_piece = action.get('piece_i') if action else None

        panel_width = self.grid_pixel_size // 3
        mini_cell = 12

        draw.text((self.grid_x, self.pieces_y - 5), "Pieces:",
                  font=self.font_small, fill=COLORS['text_white'], anchor="lb")

        for i in range(3):
            panel_x = self.grid_x + i * panel_width
            panel_y = self.pieces_y

            # Panel background
            bg_color = COLORS['combo_bg'] if i == used_piece else COLORS['grid_bg']
            draw.rounded_rectangle(
                [panel_x, panel_y, panel_x + panel_width - 10, panel_y + 70],
                radius=8, fill=bg_color
            )

            piece_id = pieces[i] if i < len(pieces) else None
            piece_color = piece_colors[i] if i < len(piece_colors) else None

            if piece_id:
                from game.pieces import PIECES_DATA
                if piece_id in PIECES_DATA:
                    shape = PIECES_DATA[piece_id]
                    color = PIECE_COLORS.get(piece_color, (150, 150, 150))

                    # Center piece in panel
                    xs = [dx for dx, dy in shape]
                    ys = [dy for dx, dy in shape]
                    w = (max(xs) - min(xs) + 1) * mini_cell
                    h = (max(ys) - min(ys) + 1) * mini_cell

                    offset_x = panel_x + (panel_width - 10 - w) // 2
                    offset_y = panel_y + (70 - h) // 2

                    for dx, dy in shape:
                        cx = offset_x + dx * mini_cell
                        cy = offset_y + dy * mini_cell
                        draw.rounded_rectangle(
                            [cx, cy, cx + mini_cell - 1, cy + mini_cell - 1],
                            radius=2, fill=color
                        )

                    # Piece ID label
                    draw.text((panel_x + panel_width // 2 - 5, panel_y + 65),
                              piece_id, font=self.font_tiny, fill=COLORS['text_white'], anchor="mb")
            else:
                draw.text((panel_x + panel_width // 2 - 5, panel_y + 35),
                          "Used", font=self.font_small, fill=(100, 100, 120), anchor="mm")

    def _draw_info_panel(self, draw: ImageDraw.Draw, info: Dict,
                         q_values_top: List[Dict] = None,
                         decision_tags: List[str] = None):
        """Draw info panel on the right side"""
        x = self.info_x
        y = self.grid_y

        # Valid moves
        num_valid = info.get('valid_mask_summary', {}).get('num_valid', 0)
        draw.text((x, y), f"Valid moves: {num_valid}",
                  font=self.font_medium, fill=COLORS['text_white'])
        y += 35

        # Clears
        k_clears = info.get('k_clears', 0)
        if k_clears > 0:
            draw.text((x, y), f"Clears: {k_clears}",
                      font=self.font_medium, fill=COLORS['text_yellow'])
            y += 30

            cleared_rows = info.get('cleared_rows', [])
            cleared_cols = info.get('cleared_cols', [])
            clear_str = ", ".join([f"R{r}" for r in cleared_rows] + [f"C{c}" for c in cleared_cols])
            draw.text((x, y), clear_str,
                      font=self.font_small, fill=COLORS['text_red'])
            y += 25

        # Reward
        reward = info.get('reward', 0)
        reward_color = COLORS['text_green'] if reward > 0 else COLORS['text_red'] if reward < 0 else COLORS['text_white']
        draw.text((x, y + 10), f"Reward: {reward:.2f}",
                  font=self.font_small, fill=reward_color)
        y += 40

        # Q-values (if provided)
        if q_values_top:
            draw.text((x, y), "Top Q-values:",
                      font=self.font_small, fill=COLORS['text_white'])
            y += 22
            for i, qv in enumerate(q_values_top[:3]):
                desc = qv.get('desc', f"a{qv.get('action_id', '?')}")
                q = qv.get('q', 0)
                draw.text((x + 10, y), f"{desc}: {q:.2f}",
                          font=self.font_tiny, fill=COLORS['text_yellow'])
                y += 18

        # Decision tags
        if decision_tags:
            y += 10
            for tag in decision_tags:
                color = COLORS['text_yellow'] if 'CLEAR' in tag.upper() else COLORS['text_white']
                draw.text((x, y), f"[{tag}]",
                          font=self.font_small, fill=color)
                y += 22

    def save_frame(self, step_info: Dict, path: str, **kwargs):
        """Render and save frame to file"""
        img = self.render_frame(step_info, **kwargs)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img.save(path)

    def render_ansi(self, step_info: Dict) -> str:
        """Render as ASCII art for console"""
        lines = []
        grid = step_info.get('grid', [[0]*8 for _ in range(8)])
        score = step_info.get('score_total', 0)
        step = step_info.get('t', 0)
        combo = step_info.get('combo_streak', 0)

        lines.append(f"Score: {score}  Step: {step}  Combo: {combo}")
        lines.append("+" + "-" * 17 + "+")

        for y in range(8):
            row = "| "
            for x in range(8):
                row += "█ " if grid[y][x] else ". "
            row += "|"
            lines.append(row)

        lines.append("+" + "-" * 17 + "+")

        # Pieces
        pieces = step_info.get('pieces', [])
        lines.append(f"Pieces: {pieces}")
        lines.append(f"Valid moves: {step_info.get('valid_mask_summary', {}).get('num_valid', 0)}")

        return "\n".join(lines)
