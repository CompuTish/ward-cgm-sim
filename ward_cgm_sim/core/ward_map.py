"""Tile-based ward layout for the top-down (Pokemon-style) view.

The map is a small grid of tiles. Beds sit in four bays of eight; the agent
walks the corridors between them. Two things depend on geometry:

* the telemetry dashboard can only be read while standing at the nurse station
  (or by spending a step on CHECK_DASHBOARD from anywhere, which models
  glancing at a handheld), and
* patient-directed actions only apply to a bed the agent is standing next to.

That is what makes movement a real cost and gives the layout clinical meaning.

Stdlib only (ships to the browser build).
"""

from collections import deque

# Tile codes
FLOOR = 0
WALL = 1
BED = 2
STATION = 3  # nurse station / telemetry dashboard
DRUG_ROOM = 4
ENTRANCE = 5  # ED / admissions doorway

WIDTH = 25
HEIGHT = 21

# Bay layout: 4 bays x 8 beds. Beds are placed in pairs of columns either side
# of a walkable corridor so every bed is reachable from an adjacent floor tile.
_BAY_ORIGINS = [(2, 2), (14, 2), (2, 12), (14, 12)]
_BED_OFFSETS = [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (3, 1), (3, 2), (3, 3)]


def _blank_grid() -> list[list[int]]:
    grid = [[FLOOR for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for x in range(WIDTH):
        grid[0][x] = WALL
        grid[HEIGHT - 1][x] = WALL
    for y in range(HEIGHT):
        grid[y][0] = WALL
        grid[y][WIDTH - 1] = WALL
    return grid


class WardMap:
    """Static ward geometry plus bed<->tile lookup."""

    def __init__(self, n_beds: int = 32):
        self.width = WIDTH
        self.height = HEIGHT
        self.n_beds = n_beds
        self.grid = _blank_grid()
        self.bed_tiles: dict[int, tuple[int, int]] = {}
        self.tile_beds: dict[tuple[int, int], int] = {}

        bed_index = 0
        for ox, oy in _BAY_ORIGINS:
            for dx, dy in _BED_OFFSETS:
                if bed_index >= n_beds:
                    break
                x, y = ox + dx, oy + dy
                self.grid[y][x] = BED
                self.bed_tiles[bed_index] = (x, y)
                self.tile_beds[(x, y)] = bed_index
                bed_index += 1

        # Nurse station with the telemetry dashboard, centre of the ward.
        self.station_tiles: list[tuple[int, int]] = []
        for x in range(10, 14):
            for y in range(9, 11):
                self.grid[y][x] = STATION
                self.station_tiles.append((x, y))

        self.drug_room_tile = (22, 9)
        self.grid[9][22] = DRUG_ROOM
        self.entrance_tile = (12, HEIGHT - 2)
        self.grid[HEIGHT - 2][12] = ENTRANCE

        self.agent_start = (12, 12)

    # ------------------------------------------------------------------
    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def tile(self, x: int, y: int) -> int:
        if not self.in_bounds(x, y):
            return WALL
        return self.grid[y][x]

    def walkable(self, x: int, y: int) -> bool:
        """Floor and the entrance are walkable; beds and furniture are not."""
        return self.tile(x, y) in (FLOOR, ENTRANCE)

    def adjacent_bed(self, x: int, y: int) -> int | None:
        """The bed the agent is standing next to, if any.

        When two beds are adjacent the lower bed number wins, deterministically.
        """
        candidates = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            bed = self.tile_beds.get((x + dx, y + dy))
            if bed is not None:
                candidates.append(bed)
        return min(candidates) if candidates else None

    def at_station(self, x: int, y: int) -> bool:
        """Standing next to (or on the edge of) the nurse station."""
        for dx, dy in ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)):
            if (x + dx, y + dy) in self.station_tiles:
                return True
        return False

    def bed_tile(self, bed: int) -> tuple[int, int]:
        return self.bed_tiles[bed]

    def approach_tile(self, bed: int) -> tuple[int, int] | None:
        """A walkable tile from which this bed can be interacted with."""
        bx, by = self.bed_tiles[bed]
        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            if self.walkable(bx + dx, by + dy):
                return (bx + dx, by + dy)
        return None

    # ------------------------------------------------------------------
    def path_length(self, start: tuple[int, int], goal: tuple[int, int]) -> int | None:
        """Breadth-first shortest walkable path length, or None if unreachable.

        Used by the rule-based agent and by staff sprites; the RL agent moves
        one tile at a time and has to learn the geometry itself.
        """
        if start == goal:
            return 0
        seen = {start}
        frontier = deque([(start, 0)])
        while frontier:
            (x, y), dist = frontier.popleft()
            for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                nxt = (x + dx, y + dy)
                if nxt in seen or not self.walkable(*nxt):
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                frontier.append((nxt, dist + 1))
        return None

    def next_step_toward(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> tuple[int, int] | None:
        """First tile on a shortest path from ``start`` to ``goal``."""
        if start == goal:
            return None
        seen = {start}
        frontier = deque()
        for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
            nxt = (start[0] + dx, start[1] + dy)
            if self.walkable(*nxt):
                seen.add(nxt)
                frontier.append((nxt, nxt))
        while frontier:
            first, current = frontier.popleft()
            if current == goal:
                return first
            for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in seen or not self.walkable(*nxt):
                    continue
                seen.add(nxt)
                frontier.append((first, nxt))
        return None
