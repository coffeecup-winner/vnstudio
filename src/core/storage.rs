use std::collections::{HashMap, hash_map};

use super::types::*;

// Ideally, this would be a generic const, but we can't do math on that in stable Rust yet
pub const CHUNK_SIZE: usize = 64;
// Size of the chunk with the external borders
const EXTENDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;
// Interval for automatic chunk deallocation
const CHUNK_DEALLOCATION_INTERVAL: u64 = 64;

#[derive(Debug, Clone)]
pub struct Chunk<State: CellState> {
    cells: [State; EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE],
}

impl<State: CellState> Default for Chunk<State> {
    fn default() -> Self {
        Self {
            cells: [State::default(); EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE],
        }
    }
}

pub trait FillNeighborhood<State: CellState, Neighborhood: CellNeighborhood<State>> {
    fn fill_neighborhood(&self, index: usize, state: &mut State, neighborhood: &mut Neighborhood);
}

impl<State: CellState> Chunk<State> {
    #[inline]
    pub fn get_start_index(&self) -> usize {
        // Skip the top border and the left border of the first row
        EXTENDED_CHUNK_SIZE + 1
    }

    #[inline]
    pub fn get_state(&self, x: usize, y: usize) -> State {
        let mut index = self.get_start_index();
        index += y * EXTENDED_CHUNK_SIZE;
        index += x;
        self.cells[index]
    }

    #[inline]
    pub fn set_state(&mut self, x: usize, y: usize, new_state: State) {
        let mut index = self.get_start_index();
        index += y * EXTENDED_CHUNK_SIZE;
        index += x;
        self.cells[index] = new_state;
    }

    #[inline]
    pub fn set_top_border(&mut self, x: usize, new_state: State) {
        self.cells[x + 1] = new_state;
    }

    #[inline]
    pub fn set_bottom_border(&mut self, x: usize, new_state: State) {
        self.cells[EXTENDED_CHUNK_SIZE * (EXTENDED_CHUNK_SIZE - 1) + x + 1] = new_state;
    }

    #[inline]
    pub fn set_left_border(&mut self, y: usize, new_state: State) {
        self.cells[EXTENDED_CHUNK_SIZE * (y + 1)] = new_state;
    }

    #[inline]
    pub fn set_right_border(&mut self, y: usize, new_state: State) {
        self.cells[EXTENDED_CHUNK_SIZE * (y + 1) + EXTENDED_CHUNK_SIZE - 1] = new_state;
    }

    #[inline]
    pub fn set_top_left_corner(&mut self, new_state: State) {
        self.cells[0] = new_state;
    }

    #[inline]
    pub fn set_top_right_corner(&mut self, new_state: State) {
        self.cells[EXTENDED_CHUNK_SIZE - 1] = new_state;
    }

    #[inline]
    pub fn set_bottom_left_corner(&mut self, new_state: State) {
        self.cells[EXTENDED_CHUNK_SIZE * (EXTENDED_CHUNK_SIZE - 1)] = new_state;
    }

    #[inline]
    pub fn set_bottom_right_corner(&mut self, new_state: State) {
        self.cells[EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE - 1] = new_state;
    }
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
        let start = index - self.get_start_index();
        neighbors[0..3].copy_from_slice(&self.cells[start..start + 3]);
        neighbors[3] = self.cells[index - 1];
        neighbors[4] = self.cells[index + 1];
        let start = start + EXTENDED_CHUNK_SIZE * 2;
        neighbors[5..8].copy_from_slice(&self.cells[start..start + 3]);

        *state = self.cells[index];
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
        neighbors[0] = self.cells[index - EXTENDED_CHUNK_SIZE];
        neighbors[1] = self.cells[index - 1];
        neighbors[2] = self.cells[index + 1];
        neighbors[3] = self.cells[index + EXTENDED_CHUNK_SIZE];

        *state = self.cells[index];
    }
}

#[derive(Clone)]
pub struct ChunkStorage<State: CellState> {
    chunks: HashMap<(isize, isize), Chunk<State>>,
    cycles_since_chunk_deallocation: u64,
}

impl<State: CellState> ChunkStorage<State> {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            cycles_since_chunk_deallocation: 0,
        }
    }

    pub fn all_chunks_iter(&self) -> hash_map::Iter<'_, (isize, isize), Chunk<State>> {
        self.chunks.iter()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
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
        if let Some(chunk) = self.chunks.get(&(chunk_x, chunk_y)) {
            chunk.get_state(cell_x, cell_y)
        } else {
            State::default()
        }
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
                let Some(chunk) = self.chunks.get(&(chunk_x, chunk_y)) else {
                    continue;
                };

                let world_min_x = chunk_x * CHUNK_SIZE as isize;
                let world_min_y = chunk_y * CHUNK_SIZE as isize;
                let local_min_x = (min.0 - world_min_x).clamp(0, CHUNK_SIZE as isize - 1) as usize;
                let local_max_x = (max.0 - world_min_x).clamp(0, CHUNK_SIZE as isize - 1) as usize;
                let local_min_y = (min.1 - world_min_y).clamp(0, CHUNK_SIZE as isize - 1) as usize;
                let local_max_y = (max.1 - world_min_y).clamp(0, CHUNK_SIZE as isize - 1) as usize;

                for cell_y in local_min_y..=local_max_y {
                    for cell_x in local_min_x..=local_max_x {
                        let state = chunk.get_state(cell_x, cell_y);
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
        let chunk = match self.chunks.entry((chunk_x, chunk_y)) {
            hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hash_map::Entry::Vacant(entry) => {
                // Don't allocate a new chunk if we're setting the cell to default state
                if new_state == State::default() {
                    return;
                }

                entry.insert(Chunk::default())
            }
        };
        chunk.set_state(cell_x, cell_y, new_state);
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
        // Set the external borders in neighboring chunks
        if cell_y == 0 {
            let top_chunk = self.chunks.entry((chunk_x, chunk_y - 1)).or_default();
            top_chunk.set_bottom_border(cell_x, new_state);

            // TODO: Only do this for Moore?
            if cell_x == 0 {
                let top_left_chunk = self.chunks.entry((chunk_x - 1, chunk_y - 1)).or_default();
                top_left_chunk.set_bottom_right_corner(new_state);
            } else if cell_x == CHUNK_SIZE - 1 {
                let top_right_chunk = self.chunks.entry((chunk_x + 1, chunk_y - 1)).or_default();
                top_right_chunk.set_bottom_left_corner(new_state);
            }
        } else if cell_y == CHUNK_SIZE - 1 {
            let bottom_chunk = self.chunks.entry((chunk_x, chunk_y + 1)).or_default();
            bottom_chunk.set_top_border(cell_x, new_state);

            // TODO: Only do this for Moore?
            if cell_x == 0 {
                let bottom_left_chunk = self.chunks.entry((chunk_x - 1, chunk_y + 1)).or_default();
                bottom_left_chunk.set_top_right_corner(new_state);
            } else if cell_x == CHUNK_SIZE - 1 {
                let bottom_right_chunk = self.chunks.entry((chunk_x + 1, chunk_y + 1)).or_default();
                bottom_right_chunk.set_top_left_corner(new_state);
            }
        }
        if cell_x == 0 {
            let left_chunk = self.chunks.entry((chunk_x - 1, chunk_y)).or_default();
            left_chunk.set_right_border(cell_y, new_state);
        } else if cell_x == CHUNK_SIZE - 1 {
            let right_chunk = self.chunks.entry((chunk_x + 1, chunk_y)).or_default();
            right_chunk.set_left_border(cell_y, new_state);
        }
    }

    pub fn set_state(&mut self, x: isize, y: isize, new_state: State) {
        let (chunk_x, cell_x) = Self::split_cell_coord(x);
        let (chunk_y, cell_y) = Self::split_cell_coord(y);
        self.set_state_core(chunk_x, chunk_y, cell_x, cell_y, new_state);
    }

    pub fn apply_changes(&mut self, chunk_changes: &[ChunkStateChanges<State>]) {
        for chunk_changes in chunk_changes {
            let (chunk_x, chunk_y) = chunk_changes.chunk_coords;
            {
                let chunk = self.chunks.entry((chunk_x, chunk_y)).or_default();
                for change in &chunk_changes.changes {
                    chunk.set_state(
                        change.cell_index_in_chunk.0,
                        change.cell_index_in_chunk.1,
                        change.new_state,
                    );
                }
            }

            for change in &chunk_changes.changes {
                self.set_neighbor_borders(
                    chunk_x,
                    chunk_y,
                    change.cell_index_in_chunk.0,
                    change.cell_index_in_chunk.1,
                    change.new_state,
                );
            }
        }
    }

    pub fn deallocate_default_chunks(&mut self) -> usize {
        let old_chunk_count = self.chunks.len();
        self.chunks
            .retain(|_, chunk| chunk.cells.iter().any(|&s| s != State::default()));
        old_chunk_count - self.chunks.len()
    }

    pub fn on_evaluate_next(&mut self) {
        self.cycles_since_chunk_deallocation += 1;
        if self.cycles_since_chunk_deallocation >= CHUNK_DEALLOCATION_INTERVAL {
            self.deallocate_default_chunks();
            self.cycles_since_chunk_deallocation = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::game_of_life::GameOfLifeState;

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
        assert_eq!(storage.chunk_count(), 0);

        storage.set_state(1, 1, GameOfLifeState::Live);

        assert_eq!(storage.chunk_count(), 1);
    }

    #[test]
    fn deallocates_fully_default_chunks() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(1, 1, GameOfLifeState::Live);
        storage.set_state(1, 1, GameOfLifeState::Dead);

        assert_eq!(storage.chunk_count(), 1);
        assert_eq!(storage.deallocate_default_chunks(), 1);
        assert_eq!(storage.chunk_count(), 0);
    }

    #[test]
    fn deallocates_default_chunks_on_configured_interval() {
        let mut storage = ChunkStorage::<GameOfLifeState>::new();
        storage.set_state(1, 1, GameOfLifeState::Live);
        storage.set_state(1, 1, GameOfLifeState::Dead);

        for _ in 0..CHUNK_DEALLOCATION_INTERVAL - 1 {
            storage.on_evaluate_next();
        }
        assert_eq!(storage.chunk_count(), 1);

        storage.on_evaluate_next();
        assert_eq!(storage.chunk_count(), 0);
    }
}
