

import pygame
import random
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass


pygame.init()






BG_COLOR = (59, 89, 152)
GRID_BG = (28, 42, 70)
GRID_LINE = (45, 65, 105)
EMPTY_CELL = (35, 50, 85)


BLOCK_COLORS = {
    'yellow': {
        'top': (255, 230, 100),
        'main': (255, 200, 50),
        'shadow': (200, 150, 30),
        'border': (180, 130, 20)
    },
    'purple': {
        'top': (200, 150, 255),
        'main': (170, 120, 230),
        'shadow': (130, 80, 180),
        'border': (100, 60, 140)
    },
    'green': {
        'top': (150, 255, 150),
        'main': (100, 220, 100),
        'shadow': (70, 170, 70),
        'border': (50, 130, 50)
    },
    'orange': {
        'top': (255, 180, 100),
        'main': (255, 140, 50),
        'shadow': (200, 100, 30),
        'border': (160, 80, 20)
    },
    'blue': {
        'top': (120, 200, 255),
        'main': (80, 160, 230),
        'shadow': (50, 120, 180),
        'border': (30, 90, 140)
    },
    'red': {
        'top': (255, 120, 120),
        'main': (230, 80, 80),
        'shadow': (180, 50, 50),
        'border': (140, 30, 30)
    },
    'cyan': {
        'top': (150, 255, 255),
        'main': (100, 220, 220),
        'shadow': (70, 170, 170),
        'border': (50, 130, 130)
    },
    'pink': {
        'top': (255, 150, 200),
        'main': (230, 120, 170),
        'shadow': (180, 80, 130),
        'border': (140, 60, 100)
    }
}


TEXT_WHITE = (255, 255, 255)
TEXT_YELLOW = (255, 220, 80)
COMBO_BLUE = (100, 180, 255)
COMBO_YELLOW = (255, 220, 50)
HIGHLIGHT_COLOR = (255, 255, 100)





BASE_WINDOW_WIDTH = 720
BASE_WINDOW_HEIGHT = 1280
BASE_UI_SCALE = 0.85
DISPLAY_MARGIN = 80

display_info = pygame.display.Info()
display_w = display_info.current_w or BASE_WINDOW_WIDTH
display_h = display_info.current_h or BASE_WINDOW_HEIGHT
fit_scale = min(
    1.0,
    (display_w - DISPLAY_MARGIN) / BASE_WINDOW_WIDTH,
    (display_h - DISPLAY_MARGIN) / BASE_WINDOW_HEIGHT,
)

WINDOW_WIDTH = int(BASE_WINDOW_WIDTH * fit_scale)
WINDOW_HEIGHT = int(BASE_WINDOW_HEIGHT * fit_scale)
UI_SCALE = min(BASE_UI_SCALE, fit_scale)

def ui(value: int) -> int:
    return int(value * UI_SCALE)
CELL_SIZE = ui(70)
GRID_SIZE = 8
GRID_PIXEL_SIZE = CELL_SIZE * GRID_SIZE
GRID_PADDING = (WINDOW_WIDTH - GRID_PIXEL_SIZE) // 2


SCORE_Y = ui(230)
BEST_SCORE_Y = ui(20)
GRID_Y = ui(380)
PIECES_Y = GRID_Y + GRID_PIXEL_SIZE + ui(80)
PIECE_PANEL_HEIGHT = ui(140)
PANEL_MARGIN_X = ui(40)
PANEL_GAP = ui(20)





PIECES_DATA = {

    '1': [(0, 0)],


    '2H': [(0, 0), (1, 0)],
    '2V': [(0, 0), (0, 1)],


    '3H': [(0, 0), (1, 0), (2, 0)],
    '3V': [(0, 0), (0, 1), (0, 2)],
    'L3_1': [(0, 0), (0, 1), (1, 1)],
    'L3_2': [(0, 0), (1, 0), (0, 1)],
    'L3_3': [(0, 0), (1, 0), (1, 1)],
    'L3_4': [(1, 0), (0, 1), (1, 1)],


    '4H': [(0, 0), (1, 0), (2, 0), (3, 0)],
    '4V': [(0, 0), (0, 1), (0, 2), (0, 3)],


    'SQ': [(0, 0), (1, 0), (0, 1), (1, 1)],


    'T_N': [(0, 0), (1, 0), (2, 0), (1, 1)],
    'T_E': [(1, 0), (0, 1), (1, 1), (1, 2)],
    'T_S': [(1, 0), (0, 1), (1, 1), (2, 1)],
    'T_W': [(0, 0), (0, 1), (1, 1), (0, 2)],


    'L_1': [(0, 0), (0, 1), (0, 2), (1, 2)],
    'L_2': [(0, 0), (1, 0), (2, 0), (0, 1)],
    'L_3': [(1, 0), (1, 1), (1, 2), (0, 2)],
    'L_4': [(2, 0), (0, 1), (1, 1), (2, 1)],
    'L_5': [(0, 0), (0, 1), (0, 2), (1, 0)],
    'L_6': [(0, 0), (1, 0), (2, 0), (2, 1)],
    'L_7': [(0, 0), (1, 0), (1, 1), (1, 2)],
    'L_8': [(0, 0), (0, 1), (1, 1), (2, 1)],


    'S4': [(1, 0), (2, 0), (0, 1), (1, 1)],
    'Z4': [(0, 0), (1, 0), (1, 1), (2, 1)],


    '5H': [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
}

@dataclass
class Piece:
    shape: List[Tuple[int, int]]
    color: str
    name: str

    def get_bounds(self):
        xs = [x for x, y in self.shape]
        ys = [y for x, y in self.shape]
        return min(xs), max(xs), min(ys), max(ys)

    def get_size(self):
        min_x, max_x, min_y, max_y = self.get_bounds()
        return max_x - min_x + 1, max_y - min_y + 1

def create_random_piece():
    name = random.choice(list(PIECES_DATA.keys()))
    shape = PIECES_DATA[name]
    color = random.choice(list(BLOCK_COLORS.keys()))
    return Piece(shape, color, name)





class BlockBlastGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Block Blast")
        self.clock = pygame.time.Clock()


        self.font_huge = pygame.font.Font(None, ui(140))
        self.font_large = pygame.font.Font(None, ui(80))
        self.font_medium = pygame.font.Font(None, ui(50))
        self.font_small = pygame.font.Font(None, ui(36))


        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.colors = [['empty' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.pieces = [create_random_piece() for _ in range(3)]

        self.score = 0
        self.best_score = 0
        self.combo_count = 0
        self.current_combo = 0

        self.selected_piece_idx = None
        self.hover_pos = None


        self.clearing_cells = []
        self.clear_animation_time = 0
        self.score_popups = []

        self.running = True





    def can_place_piece(self, piece: Piece, grid_x: int, grid_y: int) -> bool:
        for dx, dy in piece.shape:
            px, py = grid_x + dx, grid_y + dy
            if px < 0 or px >= GRID_SIZE or py < 0 or py >= GRID_SIZE:
                return False
            if self.grid[py][px] != 0:
                return False
        return True

    def place_piece(self, piece: Piece, grid_x: int, grid_y: int):
        if not self.can_place_piece(piece, grid_x, grid_y):
            return False


        for dx, dy in piece.shape:
            px, py = grid_x + dx, grid_y + dy
            self.grid[py][px] = 1
            self.colors[py][px] = piece.color

        self.last_piece = piece


        self.check_and_clear()

        return True

    def check_and_clear(self):
        lines_to_clear = []
        cols_to_clear = []


        for y in range(GRID_SIZE):
            if all(self.grid[y][x] == 1 for x in range(GRID_SIZE)):
                lines_to_clear.append(y)


        for x in range(GRID_SIZE):
            if all(self.grid[y][x] == 1 for y in range(GRID_SIZE)):
                cols_to_clear.append(x)

        if lines_to_clear or cols_to_clear:
            self.clear_lines_and_columns(lines_to_clear, cols_to_clear)

    def clear_lines_and_columns(self, lines: List[int], cols: List[int]):
        cells_to_clear = set()

        for line in lines:
            for x in range(GRID_SIZE):
                cells_to_clear.add((x, line))

        for col in cols:
            for y in range(GRID_SIZE):
                cells_to_clear.add((col, y))


        self.clearing_cells = list(cells_to_clear)
        self.clear_animation_time = 0.3


        for x, y in cells_to_clear:
            self.grid[y][x] = 0
            self.colors[y][x] = 'empty'


        total_clears = len(lines) + len(cols)
        self.current_combo = total_clears
        piece_size = len(self.last_piece.shape) if hasattr(self, 'last_piece') and self.last_piece else 0
        points = self.calculate_score(piece_size, total_clears)
        self.score += points

        if self.score > self.best_score:
            self.best_score = self.score


        self.score_popups.append({
            'text': f'+{points}',
            'x': WINDOW_WIDTH // 2,
            'y': WINDOW_HEIGHT // 2,
            'time': 1.0,
            'initial_time': 1.0
        })

    def calculate_score(self, piece_size: int, total_clears: int) -> int:
        if total_clears == 0:
            return piece_size
        elif total_clears == 1:
            return piece_size + 10
        else:
            base = total_clears * 10
            bonus = (total_clears - 1) * 20
            return piece_size + base + bonus

    def is_game_over(self) -> bool:
        for piece in self.pieces:
            if piece is None:
                continue
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    if self.can_place_piece(piece, x, y):
                        return False
        return True

    def generate_new_pieces(self):
        self.pieces = [create_random_piece() for _ in range(3)]





    def draw_3d_block(self, x: int, y: int, size: int, color_name: str, alpha: int = 255):
        colors = BLOCK_COLORS[color_name]
        radius = 0
        bevel = max(4, size // 7)
        border_width = max(1, size // 24)

        surf = pygame.Surface((size, size), pygame.SRCALPHA)


        main_rect = pygame.Rect(0, 0, size, size)
        pygame.draw.rect(surf, colors['main'], main_rect, border_radius=radius)


        top_poly = [(0, 0), (size, 0), (size - bevel, bevel), (bevel, bevel)]
        left_poly = [(0, 0), (bevel, bevel), (bevel, size - bevel), (0, size)]
        pygame.draw.polygon(surf, colors['top'], top_poly)
        pygame.draw.polygon(surf, colors['top'], left_poly)


        bottom_poly = [(0, size), (size, size), (size - bevel, size - bevel), (bevel, size - bevel)]
        right_poly = [(size, 0), (size, size), (size - bevel, size - bevel), (size - bevel, bevel)]
        pygame.draw.polygon(surf, colors['shadow'], bottom_poly)
        pygame.draw.polygon(surf, colors['shadow'], right_poly)


        pygame.draw.rect(surf, colors['border'], main_rect, width=border_width, border_radius=radius)

        if alpha < 255:
            surf.set_alpha(alpha)
        self.screen.blit(surf, (x, y))

    def draw_grid(self):

        grid_rect = pygame.Rect(GRID_PADDING, GRID_Y, GRID_PIXEL_SIZE, GRID_PIXEL_SIZE)
        pygame.draw.rect(self.screen, GRID_BG, grid_rect, border_radius=ui(20))

        cell_inset = max(2, CELL_SIZE // 14)
        cell_size_inner = CELL_SIZE - cell_inset * 2
        cell_radius = max(4, CELL_SIZE // 12)


        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                cell_x = GRID_PADDING + x * CELL_SIZE
                cell_y = GRID_Y + y * CELL_SIZE

                if self.grid[y][x] == 0:

                    pygame.draw.rect(self.screen, EMPTY_CELL,
                                   (cell_x + cell_inset, cell_y + cell_inset, cell_size_inner, cell_size_inner),
                                   border_radius=cell_radius)
                else:

                    self.draw_3d_block(cell_x, cell_y, CELL_SIZE, self.colors[y][x])


                if self.hover_pos and self.selected_piece_idx is not None:
                    piece = self.pieces[self.selected_piece_idx]
                    if piece:
                        gx, gy = self.hover_pos
                        for dx, dy in piece.shape:
                            if gx + dx == x and gy + dy == y:
                                can_place = self.can_place_piece(piece, gx, gy)
                                preview_color = (100, 255, 100, 70) if can_place else (255, 100, 100, 80)
                                surf = pygame.Surface((cell_size_inner, cell_size_inner), pygame.SRCALPHA)
                                surf.fill(preview_color)
                                self.screen.blit(surf, (cell_x + cell_inset, cell_y + cell_inset))


                if (x, y) in self.clearing_cells and self.clear_animation_time > 0:
                    alpha = int(255 * (self.clear_animation_time / 0.3))
                    surf = pygame.Surface((cell_size_inner, cell_size_inner), pygame.SRCALPHA)
                    surf.fill((255, 255, 100, alpha))
                    self.screen.blit(surf, (cell_x + cell_inset, cell_y + cell_inset))

    def draw_piece_preview(self, piece: Piece, center_x: int, center_y: int, scale: float = 1.0):
        if not piece:
            return

        block_size = int(CELL_SIZE * 0.6 * scale)
        width, height = piece.get_size()

        start_x = center_x - (width * block_size) // 2
        start_y = center_y - (height * block_size) // 2

        for dx, dy in piece.shape:
            x = start_x + dx * block_size
            y = start_y + dy * block_size
            self.draw_3d_block(x, y, block_size, piece.color)

    def draw_piece_at_pixel(
        self,
        piece: Piece,
        top_left_x: int,
        top_left_y: int,
        block_size: int,
        alpha: int = 255,
        invalid: bool = False,
    ):
        for dx, dy in piece.shape:
            x = top_left_x + dx * block_size
            y = top_left_y + dy * block_size
            self.draw_3d_block(x, y, block_size, piece.color, alpha=alpha)
            if invalid:
                overlay = pygame.Surface((block_size, block_size), pygame.SRCALPHA)
                overlay.fill((255, 80, 80, 90))
                self.screen.blit(overlay, (x, y))

    def draw_piece_on_grid(self, piece: Piece, grid_x: int, grid_y: int, alpha: int, invalid: bool):
        start_x = GRID_PADDING + grid_x * CELL_SIZE
        start_y = GRID_Y + grid_y * CELL_SIZE
        self.draw_piece_at_pixel(piece, start_x, start_y, CELL_SIZE, alpha=alpha, invalid=invalid)

    def draw_floating_piece(self):
        if self.selected_piece_idx is None:
            return
        piece = self.pieces[self.selected_piece_idx]
        if not piece:
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()

        if self.hover_pos:
            grid_x, grid_y = self.hover_pos
            can_place = self.can_place_piece(piece, grid_x, grid_y)
            alpha = 230 if can_place else 140
            self.draw_piece_on_grid(piece, grid_x, grid_y, alpha=alpha, invalid=not can_place)
        else:
            width, height = piece.get_size()
            top_left_x = mouse_x - (width * CELL_SIZE) // 2
            top_left_y = mouse_y - (height * CELL_SIZE) // 2
            self.draw_piece_at_pixel(piece, top_left_x, top_left_y, CELL_SIZE, alpha=220)

    def draw_pieces_panel(self):
        piece_width = (WINDOW_WIDTH - PANEL_MARGIN_X * 2) // 3

        for i, piece in enumerate(self.pieces):
            center_x = PANEL_MARGIN_X + piece_width // 2 + i * piece_width
            center_y = PIECES_Y + PIECE_PANEL_HEIGHT // 2


            panel_rect = pygame.Rect(
                PANEL_MARGIN_X + i * piece_width,
                PIECES_Y,
                piece_width - PANEL_GAP,
                PIECE_PANEL_HEIGHT,
            )

            panel_radius = max(8, ui(15))
            if i == self.selected_piece_idx:
                pygame.draw.rect(self.screen, (50, 70, 120), panel_rect, border_radius=panel_radius)
            else:
                pygame.draw.rect(self.screen, (35, 50, 85), panel_rect, border_radius=panel_radius)


            if piece:
                scale = 1.1 if i == self.selected_piece_idx else 1.0
                self.draw_piece_preview(piece, center_x, center_y, scale)
            else:

                text = self.font_small.render('Used', True, (100, 100, 120))
                text_rect = text.get_rect(center=(center_x, center_y))
                self.screen.blit(text, text_rect)

    def draw_ui(self):

        score_text = self.font_huge.render(str(self.score), True, TEXT_WHITE)
        score_rect = score_text.get_rect(center=(WINDOW_WIDTH // 2, SCORE_Y))


        glow_surf = pygame.Surface((score_rect.width + ui(140), score_rect.height + ui(80)), pygame.SRCALPHA)
        pygame.draw.ellipse(glow_surf, (255, 100, 200, 55), glow_surf.get_rect())
        glow_rect = glow_surf.get_rect(center=score_rect.center)
        self.screen.blit(glow_surf, glow_rect)

        score_shadow = self.font_huge.render(str(self.score), True, (20, 30, 60))
        score_shadow_rect = score_shadow.get_rect(center=(WINDOW_WIDTH // 2 + ui(3), SCORE_Y + ui(3)))
        self.screen.blit(score_shadow, score_shadow_rect)
        self.screen.blit(score_text, score_rect)


        best_text = self.font_medium.render(f"BEST {self.best_score}", True, TEXT_YELLOW)
        self.screen.blit(best_text, (ui(30), BEST_SCORE_Y))


        if self.current_combo > 1:
            combo_bg = pygame.Rect(ui(60), GRID_Y + ui(50), ui(200), ui(80))
            pygame.draw.rect(self.screen, (30, 50, 90), combo_bg, border_radius=max(8, ui(15)))

            combo_label = self.font_small.render('Combo', True, COMBO_BLUE)
            combo_value = self.font_large.render(str(self.current_combo), True, COMBO_YELLOW)

            self.screen.blit(combo_label, (ui(80), GRID_Y + ui(55)))
            self.screen.blit(combo_value, (ui(160), GRID_Y + ui(80)))

    def draw_score_popups(self):
        for popup in self.score_popups[:]:
            if popup['time'] <= 0:
                self.score_popups.remove(popup)
                continue

            alpha = int(255 * (popup['time'] / popup['initial_time']))
            offset_y = (1 - popup['time'] / popup['initial_time']) * ui(100)

            text = self.font_large.render(popup['text'], True, COMBO_YELLOW)
            text.set_alpha(alpha)
            text_rect = text.get_rect(center=(popup['x'], popup['y'] - offset_y))
            self.screen.blit(text, text_rect)

    def render(self):

        self.screen.fill(BG_COLOR)


        self.draw_grid()
        self.draw_pieces_panel()
        self.draw_floating_piece()
        self.draw_score_popups()
        self.draw_ui()

        pygame.display.flip()





    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_click(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_move(event.pos)


        mouse_pos = pygame.mouse.get_pos()
        self.update_hover(mouse_pos)

    def handle_click(self, pos):
        x, y = pos


        piece_width = (WINDOW_WIDTH - PANEL_MARGIN_X * 2) // 3
        if PIECES_Y <= y <= PIECES_Y + PIECE_PANEL_HEIGHT:
            for i in range(3):
                panel_x = PANEL_MARGIN_X + i * piece_width
                if panel_x <= x <= panel_x + piece_width - PANEL_GAP:
                    if self.pieces[i] is not None:
                        self.selected_piece_idx = i
                    return


        in_grid = GRID_PADDING <= x <= GRID_PADDING + GRID_PIXEL_SIZE and \
           GRID_Y <= y <= GRID_Y + GRID_PIXEL_SIZE
        if in_grid:
            if self.selected_piece_idx is not None:
                piece = self.pieces[self.selected_piece_idx]
                if piece:
                    grid_x = (x - GRID_PADDING) // CELL_SIZE
                    grid_y = (y - GRID_Y) // CELL_SIZE

                    if self.place_piece(piece, grid_x, grid_y):
                        self.pieces[self.selected_piece_idx] = None
                        self.selected_piece_idx = None


                        if all(p is None for p in self.pieces):
                            self.generate_new_pieces()


                        if self.is_game_over():
                            print(f"Game Over! Score: {self.score}")
                            self.reset_game()
            return


        if self.selected_piece_idx is not None:
            self.selected_piece_idx = None

    def handle_mouse_move(self, pos):
        x, y = pos


        piece_width = (WINDOW_WIDTH - PANEL_MARGIN_X * 2) // 3
        if PIECES_Y <= y <= PIECES_Y + PIECE_PANEL_HEIGHT:
            for i in range(3):
                panel_x = PANEL_MARGIN_X + i * piece_width
                if panel_x <= x <= panel_x + piece_width - PANEL_GAP:
                    if self.pieces[i] is not None:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                        return

        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

    def update_hover(self, pos):
        x, y = pos

        if GRID_PADDING <= x <= GRID_PADDING + GRID_PIXEL_SIZE and \
           GRID_Y <= y <= GRID_Y + GRID_PIXEL_SIZE and \
           self.selected_piece_idx is not None:
            grid_x = (x - GRID_PADDING) // CELL_SIZE
            grid_y = (y - GRID_Y) // CELL_SIZE
            self.hover_pos = (grid_x, grid_y)
        else:
            self.hover_pos = None

    def update(self, dt):

        if self.clear_animation_time > 0:
            self.clear_animation_time -= dt
            if self.clear_animation_time <= 0:
                self.clearing_cells = []
                self.current_combo = 0


        for popup in self.score_popups:
            popup['time'] -= dt

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.colors = [['empty' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.pieces = [create_random_piece() for _ in range(3)]
        self.score = 0
        self.current_combo = 0
        self.selected_piece_idx = None
        self.hover_pos = None
        self.clearing_cells = []
        self.score_popups = []

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0

            self.handle_events()
            self.update(dt)
            self.render()

        pygame.quit()
        sys.exit()





if __name__ == "__main__":
    game = BlockBlastGame()
    game.run()
