use std::fmt::Display;

use crate::core::types::*;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GameOfLifeState {
    #[default]
    Dead,
    Live,
}

impl TryFrom<u8> for GameOfLifeState {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => GameOfLifeState::Dead,
            1 => GameOfLifeState::Live,
            _ => return Err("Wrong cell state".to_string()),
        })
    }
}

impl From<GameOfLifeState> for u8 {
    fn from(value: GameOfLifeState) -> Self {
        match value {
            GameOfLifeState::Dead => 0,
            GameOfLifeState::Live => 1,
        }
    }
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

impl CellState for GameOfLifeState {
    const NUM_STATES: u8 = 2;
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
    type State = GameOfLifeState;
    type Evaluator = GameOfLifeEvaluator;
    type Neighborhood = MooreNeighborhood<GameOfLifeState>;
}

pub type GameOfLife = CellularAutomaton<GameOfLifeConfig>;
