use std::fmt::Debug;

use super::storage::ChunkStorage;

pub trait CellState:
    TryFrom<u8> + Into<u8> + Default + Clone + Copy + PartialEq + Eq + Debug
{
    const NUM_STATES: u8;
}

#[derive(Debug)]
pub struct Chunk<const SIZE: usize, State: CellState> {
    cells: [State; SIZE],
}

impl<const SIZE: usize, State: CellState> Default for Chunk<SIZE, State> {
    fn default() -> Self {
        Self {
            cells: [State::default(); SIZE],
        }
    }
}

pub trait CellRuleEvaluator<const NEIGHBORHOOD_SIZE: usize, State: CellState> {
    fn evaluate(cell: State, neighbors: &[State; NEIGHBORHOOD_SIZE]) -> State;
}

pub trait CellularAutomataConfig<const NEIGHBORHOOD_SIZE: usize> {
    type State: CellState;
    type Evaluator: CellRuleEvaluator<NEIGHBORHOOD_SIZE, Self::State>;
}

pub struct CellularAutomaton<
    const NEIGHBORHOOD_SIZE: usize,
    Config: CellularAutomataConfig<NEIGHBORHOOD_SIZE>,
    const CHUNK_SIZE: usize = 64,
> {
    storage: ChunkStorage<CHUNK_SIZE, Config::State>,
}

impl<
    const NEIGHBORHOOD_SIZE: usize,
    Config: CellularAutomataConfig<NEIGHBORHOOD_SIZE>,
    const CHUNK_SIZE: usize,
> CellularAutomaton<NEIGHBORHOOD_SIZE, Config, CHUNK_SIZE>
{
    pub fn new() -> Self {
        Self {
            storage: ChunkStorage::new(),
        }
    }
}

pub type VonNeumannAutomaton<Config, const CHUNK_SIZE: usize = 64> =
    CellularAutomaton<4, Config, CHUNK_SIZE>;

pub type MooreAutomaton<Config, const CHUNK_SIZE: usize = 64> =
    CellularAutomaton<8, Config, CHUNK_SIZE>;
