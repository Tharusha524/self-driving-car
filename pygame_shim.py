"""Minimal pygame shim to allow headless runs in this project.
This stub implements a tiny subset of the pygame API used by the codebase.
It is intentionally minimal and only suitable for running without rendering.
"""
import math

# Constants
QUIT = 256
SRCALPHA = 1


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def collidepoint(self, px, py):
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

    def colliderect(self, other):
        return not (self.x + self.w < other.x or other.x + other.w < self.x or
                    self.y + self.h < other.y or other.y + other.h < self.y)


class Surface:
    def __init__(self, size, flags=0):
        if isinstance(size, tuple):
            self._w, self._h = size
        else:
            # allow width-only legacy usage
            self._w = size
            self._h = size

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def blit(self, src, dest):
        return None


class _Draw:
    @staticmethod
    def rect(surface, color, rect, width=0, border_radius=0):
        return None

    @staticmethod
    def polygon(surface, color, points, width=0):
        return None

    @staticmethod
    def circle(surface, color, center, radius, width=0):
        return None

    @staticmethod
    def line(surface, color, start_pos, end_pos, width=1):
        return None


draw = _Draw()


class _Transform:
    @staticmethod
    def rotate(surface, angle):
        return surface


transform = _Transform()


class _Font:
    def __init__(self, name, size):
        self.size = size

    def render(self, text, aa, color):
        # return a dummy Surface
        return Surface((len(text) * (self.size // 2 + 1), self.size))


class font:
    @staticmethod
    def Font(name, size):
        return _Font(name, size)


class _Display:
    @staticmethod
    def set_mode(size):
        return Surface(size)

    @staticmethod
    def set_caption(title):
        return None

    @staticmethod
    def flip():
        return None


display = _Display()


class _Clock:
    def tick(self, fps):
        return None


class time:
    @staticmethod
    def Clock():
        return _Clock()


class event:
    @staticmethod
    def get():
        return []


def init():
    return None


def quit():
    return None


def Surface_from_size(size):
    return Surface(size)
