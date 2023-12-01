"""Static props in the CerealBar environment."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from environment.position import Position

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from protobuf import CerealBarProto_pb2


class TreeType(Enum):
    """ Tree type. Combination of shape and color. """
    PINE_TREE_CRV: str = "PINE_TREE_CRV"  # Pine tree with bright green leaves. Not completely straight (curvy).
    GREEN_PINE_TREE: str = "PINE_TREE"  # Light green pine tree. Completely straight.

    GREEN_ROUND_TREE: str = "TREE"  # Tree with a single bump that's green.
    TREE_RD: str = "TREE_RD"  # Same as tree, but red.
    PALM_TREE: str = "PALM_TREE"  # Palm tree.
    OAK_TREE_LM: str = "OAK_TREE_LM"  # Kind of a bouldery looking tree. Short with yellow leaves.

    PT_TREE: str = "PT_TREE"  # Similar to the green round tree; single bump, but different orientation.
    PT_TREE_LM: str = "PT_TREE_LM"  # Yellow colored PT_TREE.

    # Bush-like trees -- different small bumps.
    MULTI_TREE: str = "MULTI_TREE"  # Multicolored tree with different bumps to it. Yellow/red color.
    BUSH_TREE_BR: str = "BUSH_TREE_BR"  # Multiple bumps; brown color; shortish.
    BUSH_TREE: str = "BUSH_TREE"  # Multiple bumps; various shades of green; shortish.
    RED_BUSH_TREE: str = "BUSH_TREE_RD"  # Red bush tree.

    # Tall trees: larger bumps than the bush trees.
    TALL_TREE: str = "TALL_TREE"  # Tall green bumpy tree.
    TALL_TREE_RD: str = "TALL_TREE_RD"  # Tree with a single bump that's red.

    # "Clean" trees -- tall and skinny; conical shaped.
    CLEAN_TREE_BR: str = "CLEAN_TREE_BR"  # Tall smooth conical tree (brown color).
    CLEAN_TREE: str = "CLEAN_TREE"  # Green clean tree.
    RED_PINE_TREE: str = "CLEAN_TREE_RD"  # Red tree that's shaped like a pine tree.

    def __str__(self):
        return self.value


@dataclass
class Tree:
    """ A tree. 
    
    Attributes:
        self.tree_type: The type of tree.
        self.position: Object position.
    """
    tree_type: TreeType
    position: Position

    def add_to_buf(self, buf: CerealBarProto_pb2.MapInfo):
        prop_buf = buf.propinfo.add()
        prop_buf.pName = str(self.tree_type)
        self.position.set_buf(prop_buf.coordinate)
        prop_buf.rotV3 = '0,0,0'
