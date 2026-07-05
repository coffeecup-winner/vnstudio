use std::collections::HashMap;

use super::types::*;

// Ideally, this would be a generic const, but we can't do math on that in stable Rust yet
pub const CHUNK_SIZE: usize = 64;
// Size of the chunk with the external borders
pub const EXTENDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;
pub const EXTENDED_CHUNK_CELLS: usize = EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE;
// Interval for automatic chunk deallocation
const CHUNK_DEALLOCATION_INTERVAL: u64 = 64;

pub type Chunk<State> = [State; EXTENDED_CHUNK_CELLS];

pub trait FillNeighborhood<State: CellState, Neighborhood: CellNeighborhood<State>> {
    fn fill_neighborhood(&self, index: usize, state: &mut State, neighborhood: &mut Neighborhood);
}

#[inline]
pub fn interior_start_index() -> usize {
    // Skip the top border and the left border of the first row
    EXTENDED_CHUNK_SIZE + 1
}

#[inline]
pub fn interior_cell_index(x: usize, y: usize) -> usize {
    interior_start_index() + y * EXTENDED_CHUNK_SIZE + x
}

#[inline]
pub fn get_interior_state<State: CellState>(chunk: &Chunk<State>, x: usize, y: usize) -> State {
    chunk[interior_cell_index(x, y)]
}

#[inline]
pub fn set_interior_state<State: CellState>(
    chunk: &mut Chunk<State>,
    x: usize,
    y: usize,
    new_state: State,
) {
    chunk[interior_cell_index(x, y)] = new_state;
}

#[inline]
pub(crate) fn set_top_border<State: CellState>(
    chunk: &mut Chunk<State>,
    x: usize,
    new_state: State,
) {
    chunk[x + 1] = new_state;
}

#[inline]
pub(crate) fn set_bottom_border<State: CellState>(
    chunk: &mut Chunk<State>,
    x: usize,
    new_state: State,
) {
    chunk[EXTENDED_CHUNK_SIZE * (EXTENDED_CHUNK_SIZE - 1) + x + 1] = new_state;
}

#[inline]
pub(crate) fn set_left_border<State: CellState>(
    chunk: &mut Chunk<State>,
    y: usize,
    new_state: State,
) {
    chunk[EXTENDED_CHUNK_SIZE * (y + 1)] = new_state;
}

#[inline]
pub(crate) fn set_right_border<State: CellState>(
    chunk: &mut Chunk<State>,
    y: usize,
    new_state: State,
) {
    chunk[EXTENDED_CHUNK_SIZE * (y + 1) + EXTENDED_CHUNK_SIZE - 1] = new_state;
}

#[inline]
pub(crate) fn set_top_left_corner<State: CellState>(chunk: &mut Chunk<State>, new_state: State) {
    chunk[0] = new_state;
}

#[inline]
pub(crate) fn set_top_right_corner<State: CellState>(chunk: &mut Chunk<State>, new_state: State) {
    chunk[EXTENDED_CHUNK_SIZE - 1] = new_state;
}

#[inline]
pub(crate) fn set_bottom_left_corner<State: CellState>(chunk: &mut Chunk<State>, new_state: State) {
    chunk[EXTENDED_CHUNK_SIZE * (EXTENDED_CHUNK_SIZE - 1)] = new_state;
}

#[inline]
pub(crate) fn set_bottom_right_corner<State: CellState>(
    chunk: &mut Chunk<State>,
    new_state: State,
) {
    chunk[EXTENDED_CHUNK_CELLS - 1] = new_state;
}

fn new_chunk_cells<State: CellState>() -> Chunk<State> {
    [State::default(); EXTENDED_CHUNK_CELLS]
}

pub(crate) fn clear_halo<State: CellState>(chunk: &mut Chunk<State>) {
    for x in 0..CHUNK_SIZE {
        set_top_border(chunk, x, State::default());
        set_bottom_border(chunk, x, State::default());
    }
    for y in 0..CHUNK_SIZE {
        set_left_border(chunk, y, State::default());
        set_right_border(chunk, y, State::default());
    }
    set_top_left_corner(chunk, State::default());
    set_top_right_corner(chunk, State::default());
    set_bottom_left_corner(chunk, State::default());
    set_bottom_right_corner(chunk, State::default());
}

fn has_non_default_interior<State: CellState>(chunk: &Chunk<State>) -> bool {
    (0..CHUNK_SIZE)
        .any(|y| (0..CHUNK_SIZE).any(|x| get_interior_state(chunk, x, y) != State::default()))
}

impl<State: CellState> FillNeighborhood<State, MooreNeighborhood<State>> for Chunk<State> {
    #[inline]
    fn fill_neighborhood(
        &self,
        index: usize,
        state: &mut State,
        neighborhood: &mut MooreNeighborhood<State>,
    ) {
        // Moore neighborhood
        // 0 1 2
        // 3 X 4
        // 5 6 7
        let neighbors = neighborhood.neighbors_mut();
        let start = index - interior_start_index();
        neighbors[0..3].copy_from_slice(&self[start..start + 3]);
        neighbors[3] = self[index - 1];
        neighbors[4] = self[index + 1];
        let start = start + EXTENDED_CHUNK_SIZE * 2;
        neighbors[5..8].copy_from_slice(&self[start..start + 3]);

        *state = self[index];
    }
}

impl<State: CellState> FillNeighborhood<State, VonNeumannNeighborhood<State>> for Chunk<State> {
    #[inline]
    fn fill_neighborhood(
        &self,
        index: usize,
        state: &mut State,
        neighborhood: &mut VonNeumannNeighborhood<State>,
    ) {
        // Von Neumann neighborhood
        //   0
        // 1 X 2
        //   3
        let neighbors = neighborhood.neighbors_mut();
        neighbors[0] = self[index - EXTENDED_CHUNK_SIZE];
        neighbors[1] = self[index - 1];
        neighbors[2] = self[index + 1];
        neighbors[3] = self[index + EXTENDED_CHUNK_SIZE];

        *state = self[index];
    }
}

#[derive(Clone)]
pub struct ChunkStorage<State: CellState> {
    chunks_index: HashMap<(isize, isize), usize>,
    chunk_coords: Vec<(isize, isize)>,
    chunks: Vec<Chunk<State>>,
    next_chunk_coords: Vec<(isize, isize)>,
    next_chunks: Vec<Chunk<State>>,
    cycles_since_chunk_deallocation: u64,
}

impl<State: CellState> ChunkStorage<State> {
    pub fn new() -> Self {
        Self {
            chunks_index: HashMap::new(),
            chunk_coords: vec![],
            chunks: vec![],
            next_chunk_coords: vec![],
            next_chunks: vec![],
            cycles_since_chunk_deallocation: 0,
        }
    }

    pub fn chunks(&self) -> &[Chunk<State>] {
        &self.chunks
    }

    pub(crate) fn chunks_mut(&mut self) -> &mut [Chunk<State>] {
        &mut self.chunks
    }

    #[allow(dead_code)]
    pub fn chunk_coords(&self) -> &[(isize, isize)] {
        &self.chunk_coords
    }

    #[allow(dead_code)]
    pub fn active_cells_flat(&self) -> &[State] {
        flatten_chunk_cells(&self.chunks)
    }

    pub(crate) fn active_cells_flat_mut(&mut self) -> &mut [State] {
        flatten_chunk_cells_mut(&mut self.chunks)
    }

    #[allow(dead_code)]
    pub fn next_cells_flat_mut(&mut self) -> &mut [State] {
        flatten_chunk_cells_mut(&mut self.next_chunks)
    }

    pub fn prepare_next_chunks(&mut self) {
        self.next_chunks
            .resize_with(self.chunks.len(), new_chunk_cells);
        self.next_chunks.truncate(self.chunks.len());

        self.next_chunk_coords.clear();
        self.next_chunk_coords.extend_from_slice(&self.chunk_coords);
    }

    pub fn chunk_buffers(&mut self) -> (&[Chunk<State>], &mut [Chunk<State>]) {
        (&self.chunks, &mut self.next_chunks)
    }

    pub fn commit_next_chunks(&mut self) {
        std::mem::swap(&mut self.chunks, &mut self.next_chunks);
        std::mem::swap(&mut self.chunk_coords, &mut self.next_chunk_coords);
        self.rebuild_chunks_index();
    }

    fn split_cell_coord(coord: isize) -> (isize, usize) {
        let chunk_coord = coord.div_euclid(CHUNK_SIZE as isize);
        let cell_coord = coord.rem_euclid(CHUNK_SIZE as isize) as usize;
        (chunk_coord, cell_coord)
    }

    #[allow(dead_code)]
    pub fn get_state(&self, x: isize, y: isize) -> State {
        let (chunk_x, cell_x) = Self::split_cell_coord(x);
        let (chunk_y, cell_y) = Self::split_cell_coord(y);
        self.chunks_index
            .get(&(chunk_x, chunk_y))
            .map_or(State::default(), |&index| {
                get_interior_state(&self.chunks[index], cell_x, cell_y)
            })
    }

    fn ensure_chunk(&mut self, coords: (isize, isize)) -> usize {
        if let Some(&index) = self.chunks_index.get(&coords) {
            return index;
        }

        let index = self.chunks.len();
        self.chunk_coords.push(coords);
        self.chunks.push(new_chunk_cells());
        self.chunks_index.insert(coords, index);
        index
    }

    pub(crate) fn ensure_chunk_mut(&mut self, coords: (isize, isize)) -> &mut Chunk<State> {
        let index = self.ensure_chunk(coords);
        &mut self.chunks[index]
    }

    pub fn visit_non_default_cells(
        &self,
        min: (isize, isize),
        max: (isize, isize),
        mut visitor: impl FnMut(isize, isize, State),
    ) {
        if min.0 > max.0 || min.1 > max.1 {
            return;
        }

        let (min_chunk_x, _) = Self::split_cell_coord(min.0);
        let (max_chunk_x, _) = Self::split_cell_coord(max.0);
        let (min_chunk_y, _) = Self::split_cell_coord(min.1);
        let (max_chunk_y, _) = Self::split_cell_coord(max.1);

        for chunk_y in min_chunk_y..=max_chunk_y {
            for chunk_x in min_chunk_x..=max_chunk_x {
                let Some(&chunk_index) = self.chunks_index.get(&(chunk_x, chunk_y)) else {
                    continue;
                };
                let chunk = &self.chunks[chunk_index];

                let world_min_x = chunk_x * CHUNK_SIZE as isize;
                let world_min_y = chunk_y * CHUNK_SIZE as isize;
                let local_min_x = (min.0 - world_min_x).clamp(0, CHUNK_SIZE as isize - 1) as usize;
                let local_max_x = (max.0 - world_min_x).clamp(0, CHUNK_SIZE as isize - 1) as usize;
                let local_min_y = (min.1 - world_min_y).clamp(0, CHUNK_SIZE as isize - 1) as usize;
                let local_max_y = (max.1 - world_min_y).clamp(0, CHUNK_SIZE as isize - 1) as usize;

                for cell_y in local_min_y..=local_max_y {
                    for cell_x in local_min_x..=local_max_x {
                        let state = get_interior_state(chunk, cell_x, cell_y);
                        if state != State::default() {
                            visitor(
                                world_min_x + cell_x as isize,
                                world_min_y + cell_y as isize,
                                state,
                            );
                        }
                    }
                }
            }
        }
    }

    fn set_state_core(
        &mut self,
        chunk_x: isize,
        chunk_y: isize,
        cell_x: usize,
        cell_y: usize,
        new_state: State,
    ) {
        let coords = (chunk_x, chunk_y);
        if !self.chunks_index.contains_key(&coords) && new_state == State::default() {
            return;
        }

        set_interior_state(self.ensure_chunk_mut(coords), cell_x, cell_y, new_state);
        self.set_neighbor_borders(chunk_x, chunk_y, cell_x, cell_y, new_state);
    }

    fn set_neighbor_borders(
        &mut self,
        chunk_x: isize,
        chunk_y: isize,
        cell_x: usize,
        cell_y: usize,
        new_state: State,
    ) {
        // Set the external borders in neighboring chunks.
        if cell_y == 0 {
            let top_chunk = self.ensure_chunk_mut((chunk_x, chunk_y - 1));
            set_bottom_border(top_chunk, cell_x, new_state);

            // TODO: Only do this for Moore?
            if cell_x == 0 {
                let top_left_chunk = self.ensure_chunk_mut((chunk_x - 1, chunk_y - 1));
                set_bottom_right_corner(top_left_chunk, new_state);
            } else if cell_x == CHUNK_SIZE - 1 {
                let top_right_chunk = self.ensure_chunk_mut((chunk_x + 1, chunk_y - 1));
                set_bottom_left_corner(top_right_chunk, new_state);
            }
        } else if cell_y == CHUNK_SIZE - 1 {
            let bottom_chunk = self.ensure_chunk_mut((chunk_x, chunk_y + 1));
            set_top_border(bottom_chunk, cell_x, new_state);

            // TODO: Only do this for Moore?
            if cell_x == 0 {
                let bottom_left_chunk = self.ensure_chunk_mut((chunk_x - 1, chunk_y + 1));
                set_top_right_corner(bottom_left_chunk, new_state);
            } else if cell_x == CHUNK_SIZE - 1 {
                let bottom_right_chunk = self.ensure_chunk_mut((chunk_x + 1, chunk_y + 1));
                set_top_left_corner(bottom_right_chunk, new_state);
            }
        }
        if cell_x == 0 {
            let left_chunk = self.ensure_chunk_mut((chunk_x - 1, chunk_y));
            set_right_border(left_chunk, cell_y, new_state);
        } else if cell_x == CHUNK_SIZE - 1 {
            let right_chunk = self.ensure_chunk_mut((chunk_x + 1, chunk_y));
            set_left_border(right_chunk, cell_y, new_state);
        }
    }

    pub fn set_state(&mut self, x: isize, y: isize, new_state: State) {
        let (chunk_x, cell_x) = Self::split_cell_coord(x);
        let (chunk_y, cell_y) = Self::split_cell_coord(y);
        self.set_state_core(chunk_x, chunk_y, cell_x, cell_y, new_state);
    }

    pub fn deallocate_default_chunks(&mut self) -> usize {
        let old_chunk_count = self.chunks.len();
        let mut old_coords = std::mem::take(&mut self.chunk_coords).into_iter();
        self.chunks.retain(|chunk| {
            let keep = has_non_default_interior(chunk);
            let coords = old_coords
                .next()
                .expect("chunk coordinate count must match chunk cell count");
            if keep {
                self.chunk_coords.push(coords);
            }
            keep
        });
        debug_assert!(old_coords.next().is_none());

        let num_deallocated = old_chunk_count - self.chunks.len();
        if num_deallocated > 0 {
            self.rebuild_chunks_index();
        }
        num_deallocated
    }

    fn rebuild_chunks_index(&mut self) {
        self.chunks_index.clear();
        self.chunks_index.reserve(self.chunk_coords.len());
        for (index, &coords) in self.chunk_coords.iter().enumerate() {
            self.chunks_index.insert(coords, index);
        }
    }

    pub fn on_evaluate_next(&mut self) -> bool {
        self.cycles_since_chunk_deallocation += 1;
        if self.cycles_since_chunk_deallocation >= CHUNK_DEALLOCATION_INTERVAL {
            let deallocated = self.deallocate_default_chunks() > 0;
            self.cycles_since_chunk_deallocation = 0;
            return deallocated;
        }
        false
    }

    pub fn should_deallocate_next(&self) -> bool {
        self.cycles_since_chunk_deallocation + 1 >= CHUNK_DEALLOCATION_INTERVAL
    }
}

#[allow(dead_code)]
pub(crate) fn flatten_chunk_cells<State: CellState>(chunks: &[Chunk<State>]) -> &[State] {
    let len = chunks.len() * EXTENDED_CHUNK_CELLS;
    let ptr = chunks.as_ptr().cast::<State>();
    // SAFETY: `ChunkCells<State>` is `[State; EXTENDED_CHUNK_CELLS]`, arrays have contiguous
    // elements, and a slice of arrays is contiguous. The produced slice covers exactly the same
    // allocation and lifetime as `chunks`.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

#[allow(dead_code)]
pub(crate) fn flatten_chunk_cells_mut<State: CellState>(
    chunks: &mut [Chunk<State>],
) -> &mut [State] {
    let len = chunks.len() * EXTENDED_CHUNK_CELLS;
    let ptr = chunks.as_mut_ptr().cast::<State>();
    // SAFETY: `ChunkCells<State>` is `[State; EXTENDED_CHUNK_CELLS]`, arrays have contiguous
    // elements, and a mutable slice of arrays is contiguous and uniquely borrowed. The produced
    // slice covers exactly the same allocation and lifetime as `chunks`.
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        automata::game_of_life::{GameOfLifeEvaluator, GameOfLifeState},
        core::evaluator::BasicEvaluator,
        core::types::{CellGridEvaluator, MooreNeighborhood},
    };

    fn rebuild_halos(storage: &mut ChunkStorage<GameOfLifeState>) {
        let mut grid_evaluator = BasicEvaluator;
        <BasicEvaluator as CellGridEvaluator<
            GameOfLifeState,
            MooreNeighborhood<GameOfLifeState>,
            GameOfLifeEvaluator,
        >>::rebuild_all_halos(&mut grid_evaluator, storage);
    }

    fn evaluate_once(storage: &mut ChunkStorage<GameOfLifeState>) {
        let mut grid_evaluator = BasicEvaluator;
        let rule_evaluator = GameOfLifeEvaluator;
        storage.prepare_next_chunks();
        let chunk_coords = storage.chunk_coords().to_vec();
        let (input, output) = storage.chunk_buffers();
        grid_evaluator
            .evaluate_all(&chunk_coords, input, output, &rule_evaluator)
            .expect("basic evaluator should not fail");
        storage.commit_next_chunks();
        rebuild_halos(storage);
    }

    #[test]
    fn visits_non_default_cells_within_bounds() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(-1, -1, GameOfLifeState::Live);
        storage.set_state(63, 63, GameOfLifeState::Live);
        storage.set_state(64, 64, GameOfLifeState::Live);

        let mut visited = Vec::new();
        storage.visit_non_default_cells((-1, -1), (63, 63), |x, y, state| {
            visited.push((x, y, state));
        });
        visited.sort_by_key(|(x, y, _)| (*x, *y));

        assert_eq!(
            visited,
            vec![
                (-1, -1, GameOfLifeState::Live),
                (63, 63, GameOfLifeState::Live),
            ]
        );
    }

    #[test]
    fn reports_allocated_chunk_count() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        assert_eq!(storage.chunks().len(), 0);

        storage.set_state(1, 1, GameOfLifeState::Live);

        assert_eq!(storage.chunks().len(), 1);
    }

    #[test]
    fn deallocates_fully_default_chunks() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(1, 1, GameOfLifeState::Live);
        storage.set_state(1, 1, GameOfLifeState::Dead);

        assert_eq!(storage.chunks().len(), 1);
        assert_eq!(storage.deallocate_default_chunks(), 1);
        assert_eq!(storage.chunks().len(), 0);
    }

    #[test]
    fn deallocates_default_chunks_on_configured_interval() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(1, 1, GameOfLifeState::Live);
        storage.set_state(1, 1, GameOfLifeState::Dead);

        for _ in 0..CHUNK_DEALLOCATION_INTERVAL - 1 {
            storage.on_evaluate_next();
        }
        assert_eq!(storage.chunks().len(), 1);

        storage.on_evaluate_next();
        assert_eq!(storage.chunks().len(), 0);
    }

    #[test]
    fn rebuilds_chunk_indices_after_deallocation() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(1, 1, GameOfLifeState::Live);
        storage.set_state(65, 1, GameOfLifeState::Live);
        storage.set_state(1, 1, GameOfLifeState::Dead);

        assert_eq!(storage.deallocate_default_chunks(), 1);
        assert_eq!(storage.chunks().len(), 1);
        assert_eq!(storage.get_state(65, 1), GameOfLifeState::Live);

        storage.set_state(65, 1, GameOfLifeState::Dead);
        assert_eq!(storage.get_state(65, 1), GameOfLifeState::Dead);
    }

    #[test]
    fn flat_cells_follow_chunk_order_without_coordinate_gaps() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(0, 0, GameOfLifeState::Live);
        storage.set_state(64, 0, GameOfLifeState::Live);

        let first_chunk = storage.chunks_index[&(0, 0)];
        let second_chunk = storage.chunks_index[&(1, 0)];
        let flat = storage.active_cells_flat();

        assert_eq!(
            flat[first_chunk * EXTENDED_CHUNK_CELLS + interior_cell_index(0, 0)],
            GameOfLifeState::Live
        );
        assert_eq!(
            flat[second_chunk * EXTENDED_CHUNK_CELLS + interior_cell_index(0, 0)],
            GameOfLifeState::Live
        );
        assert_eq!(flat.len(), storage.chunks().len() * EXTENDED_CHUNK_CELLS);
    }

    #[test]
    fn next_flat_cells_mutate_output_chunk_cells() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(0, 0, GameOfLifeState::Live);
        storage.prepare_next_chunks();

        storage.next_cells_flat_mut()[interior_cell_index(1, 1)] = GameOfLifeState::Live;
        assert_eq!(
            get_interior_state(&storage.next_chunks[0], 1, 1),
            GameOfLifeState::Live
        );
    }

    #[test]
    fn next_buffer_commit_preserves_game_of_life_evolution() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(10, 9, GameOfLifeState::Live);
        storage.set_state(10, 10, GameOfLifeState::Live);
        storage.set_state(10, 11, GameOfLifeState::Live);

        evaluate_once(&mut storage);

        assert_eq!(storage.get_state(9, 10), GameOfLifeState::Live);
        assert_eq!(storage.get_state(10, 10), GameOfLifeState::Live);
        assert_eq!(storage.get_state(11, 10), GameOfLifeState::Live);
        assert_eq!(storage.get_state(10, 9), GameOfLifeState::Dead);
        assert_eq!(storage.get_state(10, 11), GameOfLifeState::Dead);
    }

    #[test]
    fn halo_rebuild_updates_edges_and_corners() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.ensure_chunk((0, 0));
        set_interior_state(&mut storage.chunks[0], 2, 0, GameOfLifeState::Live);
        set_interior_state(
            &mut storage.chunks[0],
            3,
            CHUNK_SIZE - 1,
            GameOfLifeState::Live,
        );
        set_interior_state(&mut storage.chunks[0], 0, 4, GameOfLifeState::Live);
        set_interior_state(
            &mut storage.chunks[0],
            CHUNK_SIZE - 1,
            5,
            GameOfLifeState::Live,
        );
        set_interior_state(&mut storage.chunks[0], 0, 0, GameOfLifeState::Live);
        set_interior_state(
            &mut storage.chunks[0],
            CHUNK_SIZE - 1,
            0,
            GameOfLifeState::Live,
        );
        set_interior_state(
            &mut storage.chunks[0],
            0,
            CHUNK_SIZE - 1,
            GameOfLifeState::Live,
        );
        set_interior_state(
            &mut storage.chunks[0],
            CHUNK_SIZE - 1,
            CHUNK_SIZE - 1,
            GameOfLifeState::Live,
        );

        rebuild_halos(&mut storage);

        let top = storage.chunks_index[&(0, -1)];
        let bottom = storage.chunks_index[&(0, 1)];
        let left = storage.chunks_index[&(-1, 0)];
        let right = storage.chunks_index[&(1, 0)];
        let top_left = storage.chunks_index[&(-1, -1)];
        let top_right = storage.chunks_index[&(1, -1)];
        let bottom_left = storage.chunks_index[&(-1, 1)];
        let bottom_right = storage.chunks_index[&(1, 1)];

        assert_eq!(
            storage.chunks[top][EXTENDED_CHUNK_SIZE * (EXTENDED_CHUNK_SIZE - 1) + 2 + 1],
            GameOfLifeState::Live
        );
        assert_eq!(storage.chunks[bottom][3 + 1], GameOfLifeState::Live);
        assert_eq!(
            storage.chunks[left][EXTENDED_CHUNK_SIZE * (4 + 1) + EXTENDED_CHUNK_SIZE - 1],
            GameOfLifeState::Live
        );
        assert_eq!(
            storage.chunks[right][EXTENDED_CHUNK_SIZE * (5 + 1)],
            GameOfLifeState::Live
        );
        assert_eq!(
            storage.chunks[top_left][EXTENDED_CHUNK_CELLS - 1],
            GameOfLifeState::Live
        );
        assert_eq!(
            storage.chunks[top_right][EXTENDED_CHUNK_SIZE * (EXTENDED_CHUNK_SIZE - 1)],
            GameOfLifeState::Live
        );
        assert_eq!(
            storage.chunks[bottom_left][EXTENDED_CHUNK_SIZE - 1],
            GameOfLifeState::Live
        );
        assert_eq!(storage.chunks[bottom_right][0], GameOfLifeState::Live);
    }

    #[test]
    fn edge_and_corner_interiors_allocate_required_neighbors() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.ensure_chunk((0, 0));
        set_interior_state(&mut storage.chunks[0], 0, 0, GameOfLifeState::Live);

        rebuild_halos(&mut storage);

        assert!(storage.chunks_index.contains_key(&(0, 0)));
        assert!(storage.chunks_index.contains_key(&(0, -1)));
        assert!(storage.chunks_index.contains_key(&(-1, 0)));
        assert!(storage.chunks_index.contains_key(&(-1, -1)));
    }

    #[test]
    fn deallocates_halo_only_chunks() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(0, 0, GameOfLifeState::Live);

        assert_eq!(storage.chunks().len(), 4);
        assert_eq!(storage.deallocate_default_chunks(), 3);
        assert_eq!(storage.chunks().len(), 1);
        assert_eq!(storage.get_state(0, 0), GameOfLifeState::Live);
    }

    #[test]
    fn chunk_index_map_remains_valid_after_halo_only_deallocation() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(0, 0, GameOfLifeState::Live);
        storage.set_state(128, 0, GameOfLifeState::Live);

        storage.deallocate_default_chunks();

        assert_eq!(storage.get_state(0, 0), GameOfLifeState::Live);
        assert_eq!(storage.get_state(128, 0), GameOfLifeState::Live);

        storage.set_state(128, 0, GameOfLifeState::Dead);
        assert_eq!(storage.get_state(128, 0), GameOfLifeState::Dead);
    }
}
