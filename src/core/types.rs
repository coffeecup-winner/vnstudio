use std::fmt::{Debug, Display};

use crate::core::evaluator::BasicEvaluator;
use strum::EnumCount;

use super::{
    rule_lut::RuleLUT,
    storage::{Chunk, ChunkStorage, FillNeighborhood},
};

pub trait CellState:
    TryFrom<u8>
    + Into<u8>
    + Default
    + Clone
    + Copy
    + PartialEq
    + Eq
    + Debug
    + Display
    + EnumCount
    + 'static
{
}

impl<T> CellState for T where
    T: TryFrom<u8>
        + Into<u8>
        + Default
        + Clone
        + Copy
        + PartialEq
        + Eq
        + Debug
        + Display
        + EnumCount
        + 'static
{
}

pub trait CellStateVisuals: CellState {
    fn glyph_svg(self) -> Option<&'static str>;
}

pub trait CellNeighborhood<State: CellState>: Default + Clone + Debug + 'static {
    const NUM_CELLS: u8;

    fn neighbors(&self) -> &[State];
    fn neighbors_mut(&mut self) -> &mut [State];
}

#[derive(Debug, Default, Clone)]
pub struct VonNeumannNeighborhood<State: CellState> {
    pub neighbors: [State; 4],
}

impl<State: CellState> CellNeighborhood<State> for VonNeumannNeighborhood<State> {
    const NUM_CELLS: u8 = 4;

    fn neighbors(&self) -> &[State] {
        &self.neighbors
    }

    fn neighbors_mut(&mut self) -> &mut [State] {
        &mut self.neighbors
    }
}

#[derive(Debug, Default, Clone)]
pub struct MooreNeighborhood<State: CellState> {
    pub neighbors: [State; 8],
}

impl<State: CellState> CellNeighborhood<State> for MooreNeighborhood<State> {
    const NUM_CELLS: u8 = 8;

    fn neighbors(&self) -> &[State] {
        &self.neighbors
    }

    fn neighbors_mut(&mut self) -> &mut [State] {
        &mut self.neighbors
    }
}

pub trait CellRuleEvaluator<State: CellState, Neighborhood: CellNeighborhood<State>> {
    fn evaluate(&self, state: State, neighbors: &Neighborhood) -> State;
}

pub struct CellStateChange<State: CellState> {
    pub chunk_coords: (isize, isize),
    pub cell_index_in_chunk: (usize, usize),
    #[allow(unused)]
    pub old_state: State,
    pub new_state: State,
}

pub trait CellGridEvaluator<State: CellState, Neighborhood: CellNeighborhood<State>> {
    fn evaluate_all(
        &mut self,
        storage: &ChunkStorage<State>,
        evaluator: &dyn CellRuleEvaluator<State, Neighborhood>,
    ) -> Vec<CellStateChange<State>>;
}

pub trait CellularAutomataConfig {
    type State: CellState;
    type Neighborhood: CellNeighborhood<Self::State>;
    type Evaluator: CellRuleEvaluator<Self::State, Self::Neighborhood> + Default + 'static;
}

pub struct CellularAutomaton<Config: CellularAutomataConfig> {
    storage: ChunkStorage<Config::State>,
    rule_evaluator: Box<dyn CellRuleEvaluator<Config::State, Config::Neighborhood>>,
    grid_evaluator: Box<dyn CellGridEvaluator<Config::State, Config::Neighborhood>>,
}

impl<Config: CellularAutomataConfig> CellularAutomaton<Config>
where
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    pub fn new() -> Self {
        Self {
            storage: ChunkStorage::new(),
            rule_evaluator: Box::new(Config::Evaluator::default()),
            grid_evaluator: Box::new(BasicEvaluator),
        }
    }

    pub fn get_state(&self, x: isize, y: isize) -> Config::State {
        self.storage.get_state(x, y)
    }

    pub fn set_state(&mut self, x: isize, y: isize, new_state: Config::State) {
        self.storage.set_state(x, y, new_state);
    }

    pub fn switch_to_lut(&mut self) {
        self.rule_evaluator = Box::new(RuleLUT::<Config::State, Config::Neighborhood>::compute(
            &*self.rule_evaluator,
        ));
    }

    pub fn evaluate_next(&mut self) {
        let changes = self
            .grid_evaluator
            .evaluate_all(&self.storage, &*self.rule_evaluator);
        self.storage.apply_changes(&changes);
    }
}
