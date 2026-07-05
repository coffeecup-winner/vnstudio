use std::{
    error::Error,
    fmt::{Debug, Display},
    time::Duration,
};

use crate::core::evaluator::ParallelEvaluator;
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
    + Send
    + Sync
    + CellStateVisuals
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
        + Send
        + Sync
        + CellStateVisuals
        + 'static
{
}

pub trait CellStateVisuals {
    fn glyph_svg(self) -> Option<&'static str>;
    fn pixel_color(self) -> Option<[u8; 3]>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cell<State: CellState> {
    pub x: isize,
    pub y: isize,
    pub state: State,
}

pub trait CellNeighborhood<State: CellState>:
    Default + Clone + Debug + Send + Sync + 'static
{
    const NUM_CELLS: u8;

    fn neighbors(&self) -> &[State];
    fn neighbors_mut(&mut self) -> &mut [State];
}

#[derive(Debug, Default, Clone)]
pub struct VonNeumannNeighborhood<State: CellState> {
    pub neighbors: [State; 4],
}

impl<State: CellState> VonNeumannNeighborhood<State> {
    pub fn up(&self) -> State {
        self.neighbors[0]
    }

    pub fn down(&self) -> State {
        self.neighbors[3]
    }

    pub fn left(&self) -> State {
        self.neighbors[1]
    }

    pub fn right(&self) -> State {
        self.neighbors[2]
    }
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

pub trait CellRuleEvaluator<State: CellState, Neighborhood: CellNeighborhood<State>>:
    Send + Sync
{
    fn evaluate(&self, state: State, neighbors: &Neighborhood) -> State;
}

pub trait CellGridEvaluator<
    State: CellState,
    Neighborhood: CellNeighborhood<State>,
    Evaluator: CellRuleEvaluator<State, Neighborhood> + ?Sized,
>
{
    fn evaluate_all(
        &mut self,
        chunk_coords: &[(isize, isize)],
        input: &[Chunk<State>],
        output: &mut [Chunk<State>],
        evaluator: &Evaluator,
    ) -> Result<(), Box<dyn Error>>;

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>);

    fn rebuild_all_halos_after_topology_change(&mut self, storage: &mut ChunkStorage<State>) {
        self.rebuild_all_halos(storage);
    }

    fn sync_to_host_if_needed(
        &mut self,
        _storage: &mut ChunkStorage<State>,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn storage_changed(&mut self) {}

    fn print_stats(&self) {}
}

pub trait CellularAutomataConfig {
    const NAME: &'static str;
    type State: CellState;
    type Neighborhood: CellNeighborhood<Self::State>;
    type Evaluator: CellRuleEvaluator<Self::State, Self::Neighborhood> + Default + 'static;
}

#[derive(Default)]
pub struct CellularAutomatonOperationTimes {
    pub total_grid_evaluate: Duration,
    pub total_storage_apply: Duration,
    pub total_storage_optimize: Duration,
}

pub struct CellularAutomaton<Config: CellularAutomataConfig> {
    storage: ChunkStorage<Config::State>,
    rule_evaluator: RuleLUT<Config::State, Config::Neighborhood>,
    grid_evaluator: Box<
        dyn CellGridEvaluator<
                Config::State,
                Config::Neighborhood,
                RuleLUT<Config::State, Config::Neighborhood>,
            >,
    >,
    operation_times: CellularAutomatonOperationTimes,
}

impl<Config: CellularAutomataConfig> CellularAutomaton<Config>
where
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    pub fn new() -> Self {
        let start = std::time::Instant::now();
        let rule_evaluator = RuleLUT::compute(&Config::Evaluator::default());
        println!(
            "LUT building for {} took {}ms",
            Config::NAME,
            start.elapsed().as_millis()
        );

        Self {
            storage: ChunkStorage::new(),
            rule_evaluator,
            grid_evaluator: Box::new(ParallelEvaluator::default()),
            operation_times: Default::default(),
        }
    }

    pub fn try_new_with_grid_evaluator(
        make_grid_evaluator: impl FnOnce(
            &RuleLUT<Config::State, Config::Neighborhood>,
        ) -> Result<
            Box<
                dyn CellGridEvaluator<
                        Config::State,
                        Config::Neighborhood,
                        RuleLUT<Config::State, Config::Neighborhood>,
                    >,
            >,
            Box<dyn Error>,
        >,
    ) -> Result<Self, Box<dyn Error>> {
        let start = std::time::Instant::now();
        let rule_evaluator = RuleLUT::compute(&Config::Evaluator::default());
        println!(
            "LUT building for {} took {}ms",
            Config::NAME,
            start.elapsed().as_millis()
        );
        let grid_evaluator = make_grid_evaluator(&rule_evaluator)?;

        Ok(Self {
            storage: ChunkStorage::new(),
            rule_evaluator,
            grid_evaluator,
            operation_times: Default::default(),
        })
    }

    #[allow(dead_code)]
    pub fn get_state(&mut self, x: isize, y: isize) -> Config::State {
        self.grid_evaluator
            .sync_to_host_if_needed(&mut self.storage)
            .expect("failed to synchronize cellular automaton storage");
        self.storage.get_state(x, y)
    }

    pub fn visit_non_default_cells(
        &mut self,
        min: (isize, isize),
        max: (isize, isize),
        visitor: impl FnMut(isize, isize, Config::State),
    ) {
        self.grid_evaluator
            .sync_to_host_if_needed(&mut self.storage)
            .expect("failed to synchronize cellular automaton storage");
        self.storage.visit_non_default_cells(min, max, visitor);
    }

    pub fn operation_times(&self) -> &CellularAutomatonOperationTimes {
        &self.operation_times
    }

    pub fn print_evaluator_stats(&self) {
        self.grid_evaluator.print_stats();
    }

    pub fn chunk_count(&mut self) -> usize {
        self.grid_evaluator
            .sync_to_host_if_needed(&mut self.storage)
            .expect("failed to synchronize cellular automaton storage");
        self.storage.chunks().len()
    }

    pub fn set_state(&mut self, x: isize, y: isize, new_state: Config::State) {
        self.grid_evaluator
            .sync_to_host_if_needed(&mut self.storage)
            .expect("failed to synchronize cellular automaton storage");
        self.storage.set_state(x, y, new_state);
        self.grid_evaluator.storage_changed();
    }

    #[allow(dead_code)]
    pub fn switch_to_lut(&mut self) {
        let start = std::time::Instant::now();
        self.rule_evaluator = RuleLUT::compute(&Config::Evaluator::default());
        let end = std::time::Instant::now();
        println!(
            "LUT building for {} took {}ms",
            Config::NAME,
            (end - start).as_millis()
        );
    }

    pub fn evaluate_next(&mut self) {
        let t0 = std::time::Instant::now();
        self.storage.prepare_next_chunks();
        let chunk_coords = self.storage.chunk_coords().to_vec();
        let (input, output) = self.storage.chunk_buffers();
        self.grid_evaluator
            .evaluate_all(&chunk_coords, input, output, &self.rule_evaluator)
            .expect("failed to evaluate cellular automaton grid");
        let t1 = std::time::Instant::now();
        self.storage.commit_next_chunks();
        self.grid_evaluator.rebuild_all_halos(&mut self.storage);
        let t2 = std::time::Instant::now();
        if self.storage.should_deallocate_next() {
            self.grid_evaluator
                .sync_to_host_if_needed(&mut self.storage)
                .expect("failed to synchronize cellular automaton storage before deallocation");
        }
        if self.storage.on_evaluate_next() {
            self.grid_evaluator
                .rebuild_all_halos_after_topology_change(&mut self.storage);
        }
        let t3 = std::time::Instant::now();

        self.operation_times.total_grid_evaluate += t1 - t0;
        self.operation_times.total_storage_apply += t2 - t1;
        self.operation_times.total_storage_optimize += t3 - t2;
    }
}
