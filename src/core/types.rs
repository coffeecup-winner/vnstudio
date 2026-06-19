use std::fmt::Debug;

use crate::core::rule_lut::RuleLUT;

use super::storage::ChunkStorage;

// IMPORTANT: Default state must be equal to 0u8.try_into().unwrap()
pub trait CellState:
    TryFrom<u8> + Into<u8> + Default + Clone + Copy + PartialEq + Eq + Debug + 'static
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
    fn evaluate(&self, state: State, neighbors: &[State; NEIGHBORHOOD_SIZE]) -> State;
}

pub trait CellularAutomataConfig<const NEIGHBORHOOD_SIZE: usize> {
    type State: CellState;
    type Evaluator: CellRuleEvaluator<NEIGHBORHOOD_SIZE, Self::State> + Default + 'static;
}

pub struct CellularAutomaton<
    const NEIGHBORHOOD_SIZE: usize,
    Config: CellularAutomataConfig<NEIGHBORHOOD_SIZE>,
    const CHUNK_SIZE: usize = 64,
> {
    storage: ChunkStorage<CHUNK_SIZE, Config::State>,
    evaluator: Box<dyn CellRuleEvaluator<NEIGHBORHOOD_SIZE, Config::State>>,
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
            evaluator: Box::new(Config::Evaluator::default()),
        }
    }

    pub fn switch_to_lut(&mut self) {
        self.evaluator = Box::new(RuleLUT::<NEIGHBORHOOD_SIZE, Config::State>::compute(
            &*self.evaluator,
        ));
    }
}

pub type VonNeumannAutomaton<Config, const CHUNK_SIZE: usize = 64> =
    CellularAutomaton<4, Config, CHUNK_SIZE>;

pub type MooreAutomaton<Config, const CHUNK_SIZE: usize = 64> =
    CellularAutomaton<8, Config, CHUNK_SIZE>;
