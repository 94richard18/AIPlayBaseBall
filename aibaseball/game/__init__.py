"""Game engine for the one-month plan: rules, scripted fielders, pitch-by-pitch play (CPU only)."""

from .agents import BattedBall, Pitch, StatBatter, StatPitcher, StrikeZone, Swing, TablePitcher
from .engine import Game, line_score
from .field import FieldSpec
from .fielding import FieldingSpec, GrassSpec, Play, resolve
from .state import GameState
