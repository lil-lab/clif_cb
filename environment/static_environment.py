"""Static environment information, including props and terrain."""
from __future__ import annotations

from dataclasses import dataclass
from environment.position import Position, EDGE_WIDTH
from environment.rotation import Rotation, degree_to_rotation, rotation_from_v3
from environment.terrain import OBSTACLE_TERRAINS, Terrain
from environment.static_props.hut import Hut, HutColor, HUT_PROTO_STR
from environment.static_props.plant import Plant, PlantType
from environment.static_props.structures import Lamppost, Tent, Tower, Windmill, WINDMILL_PROTO_STR, TOWER_PROTO_STR,\
    TENT_PROTO_STR, LAMPPOST_PROTO_STR
from environment.static_props.tree import Tree, TreeType

from typing import Dict, List, TYPE_CHECKING
if TYPE_CHECKING:
    from protobuf import CerealBarProto_pb2
    from typing import Set


def _terrain_to_buf(t: (Position, Terrain), buf):
    buf.coordinate.hexX = t[0].x
    buf.coordinate.hexZ = t[0].y
    buf.lType = str(t[1])


@dataclass
class StaticEnvironment:
    """Static information about an environment, including terrain and static props.
    
    Attributes:
        self.terrain: The terrain information in each location in the environment.
    
    """
    terrain: Dict[Position, Terrain]

    huts: List[Hut]
    plants: List[Plant]
    trees: List[Tree]
    windmills: List[Windmill]
    tents: List[Tent]
    towers: List[Tower]
    lampposts: List[Lamppost]

    def add_to_buf(self, map_info: CerealBarProto_pb2.MapInfo):
        for position, terrain in self.terrain.items():
            terrain_buf = map_info.cellinfo.add()
            _terrain_to_buf((position, terrain), terrain_buf)

        for hut in self.huts:
            hut.add_to_buf(map_info)

        for plant in self.plants:
            plant.add_to_buf(map_info)

        for tree in self.trees:
            tree.add_to_buf(map_info)

        for windmill in self.windmills:
            windmill.add_to_buf(map_info)

        for tent in self.tents:
            tent.add_to_buf(map_info)

        for tower in self.towers:
            tower.add_to_buf(map_info)

        for lamppost in self.lampposts:
            lamppost.add_to_buf(map_info)

    def get_obstacle_positions(self) -> Set[Position]:
        obstacle_pos: Set[Position] = set()

        for pos, ter in self.terrain.items():
            if ter in OBSTACLE_TERRAINS:
                obstacle_pos.add(pos)

        for hut in self.huts:
            obstacle_pos.add(hut.position)
        for plant in self.plants:
            obstacle_pos.add(plant.position)
        for tree in self.trees:
            obstacle_pos.add(tree.position)
        for windmill in self.windmills:
            obstacle_pos.add(windmill.position)
        for tent in self.tents:
            obstacle_pos.add(tent.position)
        for tower in self.towers:
            obstacle_pos.add(tower.position)
        for lamppost in self.lampposts:
            obstacle_pos.add(lamppost.position)

        return obstacle_pos


def static_from_proto(
        map_info: CerealBarProto_pb2.MapInfo) -> StaticEnvironment:
    huts: List[Hut] = list()
    plants: List[Plant] = list()
    trees: List[Tree] = list()
    windmills: List[Windmill] = list()
    tents: List[Tent] = list()
    towers: List[Tower] = list()
    lampposts: List[Lamppost] = list()

    for prop in map_info.propinfo:
        prop_name: str = prop.pName.replace('(Clone)',
                                            '').upper().split('\t')[0]
        position: Position = Position(prop.coordinate.hexX,
                                      prop.coordinate.hexZ)

        num_created_objects: int = 0
        # It is a tree if its name is one of the tree types.
        for tree_type in TreeType:
            if prop_name == tree_type.value:
                num_created_objects += 1
                trees.append(Tree(tree_type, position))

        # It is a plant if its name is one of the plant types.
        for plant_type in PlantType:
            if prop_name == plant_type.value:
                num_created_objects += 1
                plants.append(Plant(plant_type, position))

        # It is a hut if its name starts with the hut identifier.
        if prop_name.startswith(HUT_PROTO_STR):
            num_created_objects += 1
            rotation: Rotation = rotation_from_v3(prop.rotV3)
            huts.append(
                Hut(HutColor(prop_name.split('_')[-1]), position, rotation))
        elif prop_name.startswith(LAMPPOST_PROTO_STR):
            num_created_objects += 1
            lampposts.append(Lamppost(position))
        elif prop_name.startswith(WINDMILL_PROTO_STR):
            num_created_objects += 1
            rotation: Rotation = rotation_from_v3(prop.rotV3)
            windmills.append(Windmill(position, rotation))
        elif prop_name.startswith(TOWER_PROTO_STR):
            num_created_objects += 1
            rotation: Rotation = rotation_from_v3(prop.rotV3)
            towers.append(Tower(position, rotation))
        elif prop_name.startswith(TENT_PROTO_STR):
            num_created_objects += 1
            rotation: Rotation = rotation_from_v3(prop.rotV3)
            tents.append(Tent(position, rotation))

        if num_created_objects != 1:
            raise ValueError(f'Created {num_created_objects} from prop {prop}')

    terrains: Dict[Position, Terrain] = dict()
    for hex_cell in map_info.cellinfo:
        position: Position = Position(hex_cell.coordinate.hexX,
                                      hex_cell.coordinate.hexZ)
        terrain: Terrain = Terrain(hex_cell.lType)

        if position in terrains:
            raise ValueError(
                f'Position {position} is already in the terrain dictionary! Was {terrains[position]}; '
                f'just got {terrain}')
        terrains[position] = terrain

    expected_num_hexes: int = EDGE_WIDTH * EDGE_WIDTH
    if len(terrains) != expected_num_hexes:
        raise ValueError(
            f'Did not get {expected_num_hexes} hex terrains; got {len(terrains)} instead.'
        )

    return StaticEnvironment(terrains, huts, plants, trees, windmills, tents,
                             towers, lampposts)
