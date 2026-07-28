"""Atomic coverage of the ward geometry and its pathfinding.

`path_length` had no test at all, despite the rule-based comparator using it to
decide which patient to walk to. A comparator that mis-measures distance
quietly changes what the telemetry arm is being compared against, so this is
not merely a tidiness gap.
"""

from __future__ import annotations

import pytest

from ward_cgm_sim.core.ward_map import BED, ENTRANCE, FLOOR, STATION, WardMap


@pytest.fixture
def ward() -> WardMap:
    return WardMap()


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def test_the_ward_has_the_declared_32_beds(ward):
    assert ward.n_beds == 32
    assert len(ward.bed_tiles) == 32
    assert len(ward.tile_beds) == 32
    assert set(ward.bed_tiles) == set(range(32))


def test_a_smaller_ward_stops_placing_beds_when_it_runs_out(ward):
    small = WardMap(n_beds=6)
    assert len(small.bed_tiles) == 6
    assert set(small.bed_tiles) == set(range(6))


def test_bed_tiles_and_tile_beds_are_the_same_mapping_both_ways(ward):
    for bed, tile in ward.bed_tiles.items():
        assert ward.tile_beds[tile] == bed
        assert ward.tile(*tile) == BED


def test_the_ward_is_walled_all_the_way_round(ward):
    for x in range(ward.width):
        assert not ward.walkable(x, 0)
        assert not ward.walkable(x, ward.height - 1)
    for y in range(ward.height):
        assert not ward.walkable(0, y)
        assert not ward.walkable(ward.width - 1, y)


def test_out_of_bounds_reads_as_wall_rather_than_raising(ward):
    assert not ward.in_bounds(-1, 0)
    assert not ward.in_bounds(0, -1)
    assert not ward.in_bounds(ward.width, 0)
    assert not ward.in_bounds(0, ward.height)
    assert not ward.walkable(-5, -5)
    assert not ward.walkable(999, 999)


def test_only_floor_and_the_doorway_are_walkable(ward):
    for y in range(ward.height):
        for x in range(ward.width):
            walkable = ward.walkable(x, y)
            assert walkable == (ward.tile(x, y) in (FLOOR, ENTRANCE))
            if ward.tile(x, y) in (BED, STATION):
                assert not walkable, "furniture must block"


def test_every_bed_can_be_reached_from_somewhere(ward):
    """A bed with no approach tile could never be acted on."""
    for bed in ward.bed_tiles:
        approach = ward.approach_tile(bed)
        assert approach is not None, f"bed {bed} is unreachable"
        assert ward.walkable(*approach)
        assert ward.adjacent_bed(*approach) is not None


def test_adjacent_bed_finds_only_orthogonal_neighbours(ward):
    bed = 0
    bx, by = ward.bed_tile(bed)
    assert ward.adjacent_bed(bx, by) is None or ward.tile(bx, by) == BED
    approach = ward.approach_tile(bed)
    assert ward.adjacent_bed(*approach) is not None
    # A diagonal neighbour is not adjacent.
    diagonal = (bx + 1, by + 1)
    if ward.walkable(*diagonal):
        beds = {
            ward.tile_beds.get((diagonal[0] + dx, diagonal[1] + dy))
            for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0))
        }
        assert ward.adjacent_bed(*diagonal) in beds


def test_two_adjacent_beds_resolve_to_the_lower_number_deterministically(ward):
    """Documented tie-break. The stock layout never puts a tile between two
    beds, so construct the case rather than skip it - an untested tie-break is
    how a patient-directed action starts landing on the wrong patient."""
    tile = (1, 1)
    assert ward.walkable(*tile)
    assert ward.adjacent_bed(*tile) is None, "positive control: no bed here yet"

    ward.tile_beds[(tile[0], tile[1] - 1)] = 9
    ward.tile_beds[(tile[0] + 1, tile[1])] = 4
    assert ward.adjacent_bed(*tile) == 4

    ward.tile_beds[(tile[0], tile[1] + 1)] = 2
    assert ward.adjacent_bed(*tile) == 2, "the lowest bed number must win"


def test_the_station_is_reachable_and_reads_as_at_station(ward):
    assert ward.station_tiles
    touching = [
        (x, y)
        for y in range(ward.height)
        for x in range(ward.width)
        if ward.walkable(x, y) and ward.at_station(x, y)
    ]
    assert touching, "the board cannot be reached on foot"


def test_a_tile_far_from_the_station_is_not_at_station(ward):
    corner = (1, 1)
    assert ward.walkable(*corner)
    assert not ward.at_station(*corner)


# --------------------------------------------------------------------------
# Pathfinding
# --------------------------------------------------------------------------


def test_the_distance_to_where_you_already_stand_is_zero(ward):
    assert ward.path_length((1, 1), (1, 1)) == 0


def test_the_distance_to_the_next_tile_is_one(ward):
    start = (1, 1)
    for dx, dy in ((0, 1), (0, -1), (-1, 0), (1, 0)):
        goal = (start[0] + dx, start[1] + dy)
        if ward.walkable(*goal):
            assert ward.path_length(start, goal) == 1
            return
    raise AssertionError("positive control: (1, 1) must have a walkable neighbour")


def test_path_length_is_the_shortest_route_not_just_any_route(ward):
    """Manhattan distance is a lower bound; the answer may exceed it around
    furniture, but it must never be shorter."""
    starts = [(1, 1), (12, 12), (2, 8)]
    goals = [(ward.width - 2, ward.height - 2), (12, 6), (20, 18)]
    checked = 0
    for start in starts:
        for goal in goals:
            if not (ward.walkable(*start) and ward.walkable(*goal)):
                continue
            distance = ward.path_length(start, goal)
            assert distance is not None, f"{start} -> {goal} should be reachable"
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            assert distance >= manhattan
            assert (distance - manhattan) % 2 == 0, "parity must be preserved"
            checked += 1
    assert checked >= 3, "positive control: pairs must actually have been measured"


def test_path_length_is_symmetric(ward):
    a, b = (1, 1), (20, 18)
    assert ward.walkable(*a) and ward.walkable(*b)
    assert ward.path_length(a, b) == ward.path_length(b, a)


def test_an_unreachable_goal_returns_none_rather_than_a_number(ward):
    wall = (0, 0)
    assert not ward.walkable(*wall)
    assert ward.path_length((1, 1), wall) is None


def test_next_step_toward_moves_one_tile_closer_every_time(ward):
    """Walk the whole route and confirm it terminates at the goal."""
    start, goal = (1, 1), (20, 18)
    distance = ward.path_length(start, goal)
    assert distance is not None and distance > 5, "positive control: a real journey"

    position, taken = start, 0
    while position != goal:
        step = ward.next_step_toward(position, goal)
        assert step is not None, f"stuck at {position}"
        assert ward.walkable(*step)
        assert abs(step[0] - position[0]) + abs(step[1] - position[1]) == 1
        position, taken = step, taken + 1
        assert taken <= distance, "the route is longer than the shortest path"
    assert taken == distance


def test_there_is_no_step_toward_where_you_already_are(ward):
    assert ward.next_step_toward((1, 1), (1, 1)) is None


def test_there_is_no_step_toward_somewhere_unreachable(ward):
    assert ward.next_step_toward((1, 1), (0, 0)) is None


def test_every_walkable_tile_can_reach_every_other_one(ward):
    """One sealed-off pocket would strand the agent or a colleague there."""
    walkable = [
        (x, y)
        for y in range(ward.height)
        for x in range(ward.width)
        if ward.walkable(x, y)
    ]
    assert len(walkable) > 100, "positive control: the ward must have floor"
    origin = walkable[0]
    unreachable = [t for t in walkable if ward.path_length(origin, t) is None]
    assert not unreachable, f"{len(unreachable)} tiles are cut off, e.g. {unreachable[:3]}"
