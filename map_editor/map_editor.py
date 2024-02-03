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
CIRCLE_RADIUS = 5
SCALE_BACKGROUND_ALPHA = (255, 255, 255, 128)
SCALE_LINE_ALPHA = (0, 0, 0, 128)


@dataclass
class MapEditorUI:
    horizontal_bounds: MapBound = (0, 100)
    vertical_bounds: MapBound = (0, 100)
    scaling: float = 10
    width: float = 1200
    height: float = 1000
    draw_polygon: bool = True
    draw_rectangle: bool = False
    new_outline: List[Vec2D] = field(default_factory=list)
    screen: pygame.Surface = field(init=False)
    scaling_bg: pygame.Surface = field(init=False)
    is_exit_requested: bool = False
    ui_events_thread: Thread = field(init=False)
    polygons: List[List[Vec2D]] = field(default_factory=list)

    def __post_init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption('PySF Map Editor')
        self.scaling_bg = self._prerender_scale()
        self.screen.fill(COLOR_BACKGROUND)
        pygame.display.update()

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
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    if self.is_hover_over_start:
                        self.polygons.append(self.new_outline)
                        self.new_outline = []
                    elif self.is_append_pos_active:
                        self.new_outline.append(pos)

    def _prerender_scale(self) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surface.fill(SCALE_BACKGROUND_ALPHA)

        for i in range(int(self.width / self.scaling)):
            x = i * self.scaling
            p1, p2 = (x, 0), (x, self.height)
            pygame.draw.line(surface, SCALE_LINE_ALPHA, p1, p2)

        for i in range(int(self.height / self.scaling)):
            y = i * self.scaling
            p1, p2 = (0, y), (self.width, y)
            pygame.draw.line(surface, SCALE_LINE_ALPHA, p1, p2)

        return surface

    def _render_screen(self):
        sleep(0.01)

        if self.is_exit_requested:
            print(self.polygons)
            pygame.quit()
            self.ui_events_thread.join()
            exit()

        self.screen.fill(COLOR_BACKGROUND)
        self.screen.blit(self.scaling_bg, (0, 0))

        for poly in self.polygons:
            pygame.draw.polygon(self.screen, COLOR_POLYGON, poly)

        for p in self.new_outline:
            pygame.draw.circle(self.screen, COLOR_NEW_POINT, p, CIRCLE_RADIUS)

        if self.is_hover_over_start:
            pygame.draw.circle(self.screen, COLOR_NEW_POINT_HOVER, self.new_outline[0], CIRCLE_RADIUS)

        for p1, p2 in zip(self.new_outline[:-1], self.new_outline[1:]):
            pygame.draw.line(self.screen, COLOR_NEW_POINT, p1, p2)

        pygame.display.update()


def main():
    ui = MapEditorUI()
    ui.show()


if __name__ == "__main__":
    main()
