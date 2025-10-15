import sys
import heapq
from typing import Tuple, List, Dict

TYPES = ('A', 'B', 'C', 'D')
COST = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
TARGET_ROOM = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
ENTRANCES = (2, 4, 6, 8)
HALLWAY_STOPS = (0, 1, 3, 5, 7, 9, 10)


State = Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]

def parse_input(lines: List[str]) -> State:
    hallway = None
    letter_rows: List[List[str]] = []

    for raw in lines:
        line = raw.rstrip('\n')
        if line.startswith('#') and line.endswith('#'):
            core = line[1:-1]
            if len(core) == 11 and all(c in '.ABCD' for c in core):
                hallway = tuple(core)

        letters = [c for c in line if c in TYPES]
        if len(letters) == 4:
            letter_rows.append(letters)

    if hallway is None:
        hallway = tuple('.' for _ in range(11))

    depth = len(letter_rows)

    rooms: List[Tuple[str, ...]] = []
    for r in range(4):
        col = tuple(letter_rows[d][r] for d in range(depth))
        rooms.append(col)

    return (hallway, tuple(rooms))


def is_goal(state: State) -> bool:
    hallway, rooms = state
    if any(c != '.' for c in hallway):
        return False
    for r, room in enumerate(rooms):
        if any(c != TYPES[r] for c in room):
            return False
    return True


def clear_hallway(hallway: Tuple[str, ...], start: int, end: int) -> bool:
    step = 1 if end > start else -1
    p = start + step
    while True:
        if hallway[p] != '.':
            return False
        if p == end:
            break
        p += step
    return True


def room_ready_for(letter: str, room: Tuple[str, ...]) -> bool:
    return all(c in ('.', letter) for c in room)


def top_occupant_index(room: Tuple[str, ...]) -> int:
    for i, c in enumerate(room):
        if c != '.':
            return i
    return -1


def deepest_empty_index(room: Tuple[str, ...]) -> int:
    for i in range(len(room) - 1, -1, -1):
        if room[i] == '.':
            return i
    return -1


def room_settled(room_index: int, room: Tuple[str, ...]) -> bool:
    need = TYPES[room_index]
    return all(c in ('.', need) for c in room) and all(c == need for c in room if c != '.')


def neighbors(state: State) -> List[Tuple[State, int]]:
    hallway, rooms = state
    res: List[Tuple[State, int]] = []

    depth = len(rooms[0])

    for hp, who in enumerate(hallway):
        if who == '.':
            continue
        target_r = TARGET_ROOM[who]
        entrance = ENTRANCES[target_r]
        if not clear_hallway(hallway, hp, entrance):
            continue
        room = rooms[target_r]
        if not room_ready_for(who, room):
            continue
        dest_i = deepest_empty_index(room)
        if dest_i == -1:
            continue
        steps = abs(hp - entrance) + (dest_i + 1)
        cost = steps * COST[who]

        new_hallway = list(hallway)
        new_hallway[hp] = '.'
        new_room = list(room)
        new_room[dest_i] = who

        new_rooms = list(rooms)
        new_rooms[target_r] = tuple(new_room)

        res.append(((tuple(new_hallway), tuple(new_rooms)), cost))

    for r_index, room in enumerate(rooms):
        entrance = ENTRANCES[r_index]

        if hallway[entrance] != '.':
            continue

        ti = top_occupant_index(room)
        if ti == -1:
            continue

        who = room[ti]

        if TARGET_ROOM[who] == r_index and all(c in ('.', who) for c in room[ti:]):
            continue

        for hp in HALLWAY_STOPS:
            if clear_hallway(hallway, entrance, hp):
                steps = (ti + 1) + abs(hp - entrance)
                cost = steps * COST[who]

                new_hallway = list(hallway)
                new_hallway[hp] = who
                new_room = list(room)
                new_room[ti] = '.'

                new_rooms = list(rooms)
                new_rooms[r_index] = tuple(new_room)

                res.append(((tuple(new_hallway), tuple(new_rooms)), cost))

    return res


def dijkstra(start: State) -> int:
    pq: List[Tuple[int, State]] = []
    heapq.heappush(pq, (0, start))
    best: Dict[State, int] = {start: 0}

    while pq:
        cost, state = heapq.heappop(pq)
        if cost != best.get(state, float('inf')):
            continue
        if is_goal(state):
            return cost
        for nxt, w in neighbors(state):
            nc = cost + w
            if nc < best.get(nxt, float('inf')):
                best[nxt] = nc
                heapq.heappush(pq, (nc, nxt))

def main():
    lines = [line.rstrip('\n') for line in sys.stdin if line.strip() != '']
    start = parse_input(lines)
    print(dijkstra(start))


if __name__ == "__main__":
    main()
