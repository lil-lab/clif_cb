"""Positions in CerealBar."""
from __future__ import annotations

import numpy as np

from dataclasses import dataclass

EDGE_WIDTH: int = 25

even_x_positions = np.linspace(0, (EDGE_WIDTH - 1) * 2, EDGE_WIDTH)
y_positions = np.linspace(0, EDGE_WIDTH - 1, EDGE_WIDTH)
coords = np.array(np.meshgrid(even_x_positions, y_positions)).transpose(
    (1, 2, 0))

# This turns the coordinates into the double-width format.
for i in range(EDGE_WIDTH):
    if i % 2 == 1:
        for j in range(EDGE_WIDTH):
            coords[i][j][0] += 1

flat_coords = np.reshape(coords, [-1, 2])
tiled = np.tile(flat_coords[:, np.newaxis, :], [1, flat_coords.shape[0], 1])
transposed_tiled = tiled.transpose((1, 0, 2))
DISTANCE_MATRIX: np.array = np.array(np.linalg.norm(
    tiled / (EDGE_WIDTH / 5) - transposed_tiled / (EDGE_WIDTH / 5),
    ord=1,
    axis=2),
                                     order='C') / 2


@dataclass
class Position:
    """An (x, y) (offset) coordinate in CerealBar.
    
    Attributes:
        self.x: The (offset) x-coordinate.
        self.y: The (offset) y-coordinate.
    """
    x: int
    y: int

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False

        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __hash__(self):
        return (self.x, self.y).__hash__()

    def __lt__(self, other):
        return str(self) < str(other)

    def set_buf(self, buf):
        buf.hexX = self.x
        buf.hexZ = self.y


def out_of_bounds(pos: Position):
    return pos.x < 0 or pos.y < 0 or pos.x >= EDGE_WIDTH or pos.y >= EDGE_WIDTH


def compute_distance(a: Position, b: Position) -> int:
    pos_a = EDGE_WIDTH * a.x + a.y
    pos_b = EDGE_WIDTH * b.x + b.y

    distance = DISTANCE_MATRIX[pos_a][pos_b]
    return distance * 5
