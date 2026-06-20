use std::fmt::Display;

use crate::core::types::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use strum::EnumCount;

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, IntoPrimitive, TryFromPrimitive, EnumCount,
)]
#[repr(u8)]
pub enum GameOfLifeState {
    #[default]
    Dead = 0,
    Live = 1,
}

impl Display for GameOfLifeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let char = match &self {
            GameOfLifeState::Dead => "_",
            GameOfLifeState::Live => "O",
        };
        f.write_str(char)
    }
}

impl CellStateVisuals for GameOfLifeState {
    fn glyph_svg(self) -> Option<&'static str> {
        match self {
            GameOfLifeState::Dead => None,
            GameOfLifeState::Live => Some(include_str!("glyphs/live.svg")),
        }
    }

    fn pixel_color(self) -> Option<[u8; 3]> {
        match self {
            GameOfLifeState::Dead => None,
            GameOfLifeState::Live => Some([32, 33, 36]),
        }
    }
}

#[derive(Default)]
pub struct GameOfLifeEvaluator;

impl CellRuleEvaluator<GameOfLifeState, MooreNeighborhood<GameOfLifeState>>
    for GameOfLifeEvaluator
{
    fn evaluate(
        &self,
        state: GameOfLifeState,
        neighbors: &MooreNeighborhood<GameOfLifeState>,
    ) -> GameOfLifeState {
        let num_live_neighbors = neighbors
            .neighbors()
            .iter()
            .filter(|s| **s == GameOfLifeState::Live)
            .count();
        if state == GameOfLifeState::Live {
            match num_live_neighbors {
                2 | 3 => GameOfLifeState::Live,
                _ => GameOfLifeState::Dead,
            }
        } else {
            if num_live_neighbors == 3 {
                GameOfLifeState::Live
            } else {
                GameOfLifeState::Dead
            }
        }
    }
}

pub struct GameOfLifeConfig;

impl CellularAutomataConfig for GameOfLifeConfig {
    const NAME: &'static str = "Game of Life";
    type State = GameOfLifeState;
    type Evaluator = GameOfLifeEvaluator;
    type Neighborhood = MooreNeighborhood<GameOfLifeState>;
}

pub type GameOfLife = CellularAutomaton<GameOfLifeConfig>;
