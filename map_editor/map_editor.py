from math import dist
from typing import Tuple, List
from dataclasses import dataclass, field
from threading import Thread
from signal import signal, SIGINT
from time import sleep
import pygame

Vec2D = Tuple[float, float]
MapBound = Tuple[Vec2D, Vec2D]

COLOR_NEW_POINT = (255, 120, 120)
COLOR_NEW_POINT_HOVER = (60, 120, 180)
COLOR_BACKGROUND = (255, 255, 255)
COLOR_POLYGON = (30, 30, 30)
COLOR_RECT = (120, 30, 30)
CIRCLE_RADIUS = 5
SCALE_BACKGROUND_ALPHA = (255, 255, 255, 128)
SCALE_LINE_ALPHA = (0, 0, 0, 128)


@dataclass
class MapEditorUI:
    horizontal_bounds: MapBound = (0, 100)
    vertical_bounds: MapBound = (0, 100)
    scaling: float = 1
    x_offset: float = 0
    y_offset: float = 0
    raster_px: int = 10
    width: float = 1200
    height: float = 1000
    draw_polygon: bool = False
    draw_rectangle: bool = True
    new_outline: List[Vec2D] = field(default_factory=list)
    screen: pygame.Surface = field(init=False)
    scaling_bg: pygame.Surface = field(init=False)
    scaling_changed: bool = False
    is_exit_requested: bool = False
    ui_events_thread: Thread = field(init=False)
    polygons: List[List[Vec2D]] = field(default_factory=list)
    rectangles: List[List[Vec2D]] = field(default_factory=list)

    def __post_init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption('PySF Map Editor')
        self.scaling_bg = self._prerender_scale()
        self.screen.fill(COLOR_BACKGROUND)
        pygame.display.update()
        self.polygons = [
            [(615, 312), (744, 411), (612, 587), (435, 540), (385, 401), (471, 322), (608, 311)],
            [(933, 344), (983, 509), (899, 665), (808, 751), (883, 817), (1070, 839), (1114, 722),
             (1143, 540), (1147, 336), (1122, 134), (1089, 59), (971, 57), (861, 187)],
            [(506, 206), (392, 222), (280, 310), (215, 401), (128, 380), (43, 378),
             (44, 124), (58, 44), (207, 43), (364, 54), (503, 90)],
            [(101, 893), (340, 898), (575, 888), (582, 860), (522, 752), (412, 684),
             (316, 623), (174, 654), (109, 747), (101, 887), (101, 899)]
        ]
        self.rectangles = [
            [(778, 478), (680, 600), (783.11, 682.82), (881.11, 560.82)],
            [(78, 461), (72, 618), (231.12, 624.08), (237.12, 467.08)],
            [(597, 53), (589, 178), (785.13, 190.55), (793.13, 65.55)]
        ]

    @property
    def is_append_pos_active(self) -> bool:
        return self.draw_polygon or self.draw_rectangle

    @property
    def is_hover_over_start(self) -> bool:
        return len(self.new_outline) > 0 and \
            dist(pygame.mouse.get_pos(), self.new_outline[0]) <= CIRCLE_RADIUS

    def show(self):
        self.ui_events_thread = Thread(target=self._process_event_queue)
        self.ui_events_thread.start()

        def handle_sigint(signum, frame):
            self.is_exit_requested = True

        signal(SIGINT, handle_sigint)

        while True:
            self._render_screen()

    def _process_event_queue(self):
        while not self.is_exit_requested:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.is_exit_requested = True
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    pos = pygame.mouse.get_pos()
                    if self.draw_polygon:
                        if self.is_hover_over_start:
                            self.polygons.append(self.new_outline)
                            self.new_outline = []
                        else:
                            self.new_outline.append(pos)
                    elif self.draw_rectangle:
                        if len(self.new_outline) < 2:
                            self.new_outline.append(pos)
                        else:
                            p1, p2 = self.new_outline[0], self.new_outline[1]
                            _, _, p3, p4 = rect_of(p1, p2, pos)
                            self.rectangles.append([p1, p2, p3, p4])
                            self.new_outline = []
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                    self.new_outline = []
                elif event.type == pygame.MOUSEWHEEL and event.y != 0:
                    self.scaling_changed = True
                    if event.y == -1:
                        self.scaling += 0.1
                    else:
                        self.scaling = max(0.1, self.scaling - 0.1)
                    # TODO: update x / y offsets such that the point at the
                    #       mouse position stays there after transformation

    def _prerender_scale(self) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surface.fill(SCALE_BACKGROUND_ALPHA)
        raster_scaling = self.scaling * self.raster_px

        for i in range(int(self.width / raster_scaling)):
            x = i * raster_scaling
            p1, p2 = (x, 0), (x, self.height)
            pygame.draw.line(surface, SCALE_LINE_ALPHA, p1, p2)

        for i in range(int(self.height / raster_scaling)):
            y = i * raster_scaling
            p1, p2 = (0, y), (self.width, y)
            pygame.draw.line(surface, SCALE_LINE_ALPHA, p1, p2)

        return surface

    def _render_screen(self):
        sleep(0.01)

        if self.is_exit_requested:
            print("polygons:", self.polygons)
            print("rects:", self.rectangles)
            pygame.quit()
            self.ui_events_thread.join()
            exit()

        if self.scaling_changed:
            self.scaling_bg = self._prerender_scale()
            self.scaling_changed = False

        self.screen.fill(COLOR_BACKGROUND)
        self.screen.blit(self.scaling_bg, (0, 0))

        def scale_point(p: Vec2D) -> Vec2D:
            return p[0] * self.scaling + self.x_offset, \
                p[1] * self.scaling + self.y_offset

        def unscale_point(p: Vec2D) -> Vec2D:
            return p[0] - self.x_offset / self.scaling, \
                p[1] - self.y_offset / self.scaling

        for poly in self.polygons:
            scaled_poly = [scale_point(p) for p in poly]
            pygame.draw.polygon(self.screen, COLOR_POLYGON, scaled_poly)

        for rect in self.rectangles:
            scaled_rect = [scale_point(p) for p in rect]
            pygame.draw.polygon(self.screen, COLOR_RECT, scaled_rect)

        for p in self.new_outline:
            scaled_p = scale_point(p)
            pygame.draw.circle(self.screen, COLOR_NEW_POINT, scaled_p, CIRCLE_RADIUS)

        if self.is_hover_over_start:
            scaled_p = scale_point(self.new_outline[0])
            pygame.draw.circle(self.screen, COLOR_NEW_POINT_HOVER, scaled_p, CIRCLE_RADIUS)

        for p1, p2 in zip(self.new_outline[:-1], self.new_outline[1:]):
            scaled_p1, scaled_p2 = scale_point(p1), scale_point(p2)
            pygame.draw.line(self.screen, COLOR_NEW_POINT, scaled_p1, scaled_p2)

        if self.draw_rectangle and len(self.new_outline) == 2:
            p1, p2 = self.new_outline[0], self.new_outline[1]
            pos = unscale_point(pygame.mouse.get_pos())
            _, _, p3, p4 = rect_of(p1, p2, pos)
            scaled_p1, scaled_p2 = scale_point(p1), scale_point(p2)
            scaled_p3, scaled_p4 = scale_point(p3), scale_point(p4)
            pygame.draw.line(self.screen, COLOR_NEW_POINT, scaled_p2, scaled_p3)
            pygame.draw.line(self.screen, COLOR_NEW_POINT, scaled_p3, scaled_p4)
            pygame.draw.line(self.screen, COLOR_NEW_POINT, scaled_p4, scaled_p1)

        pygame.display.update()


def rect_of(p1: Vec2D, p2: Vec2D, p3: Vec2D) -> Tuple[Vec2D, Vec2D, Vec2D, Vec2D]:
    # with p1, p2 as base and p3 for orthogonal distance
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    ox, oy = y2 - y1, x1 - x2
    x4, y4 = x3 + ox, y3 + oy

    num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = num / den
    cx, cy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    dx, dy = x3 - cx, y3 - cy
    p3, p4 = (x2 + dx, y2 + dy), (x1 + dx, y1 + dy)
    return p1, p2, p3, p4


def main():
    ui = MapEditorUI()
    ui.show()


if __name__ == "__main__":
    main()
