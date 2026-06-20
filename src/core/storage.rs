use std::collections::{HashMap, hash_map};

use super::types::*;

// Ideally, this would be a generic const, but we can't do math on that in stable Rust yet
pub const CHUNK_SIZE: usize = 64;
// Size of the chunk with the external borders
pub const EXTENDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;

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
}

impl<State: CellState> ChunkStorage<State> {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }

    pub fn all_chunks_iter(&self) -> hash_map::Iter<'_, (isize, isize), Chunk<State>> {
        self.chunks.iter()
    }

    fn split_cell_coord(coord: isize) -> (isize, usize) {
        let chunk_coord = coord.div_euclid(CHUNK_SIZE as isize);
        let cell_coord = coord.rem_euclid(CHUNK_SIZE as isize) as usize;
        (chunk_coord, cell_coord)
    }

    pub fn get_state(&self, x: isize, y: isize) -> State {
        let (chunk_x, cell_x) = Self::split_cell_coord(x);
        let (chunk_y, cell_y) = Self::split_cell_coord(y);
        if let Some(chunk) = self.chunks.get(&(chunk_x, chunk_y)) {
            chunk.get_state(cell_x, cell_y)
        } else {
            State::default()
        }
    }

    pub fn set_state(&mut self, x: isize, y: isize, new_state: State) {
        // TODO
    }

    pub fn apply_changes(&mut self, changes: &[CellStateChange<State>]) {
        for change in changes {
            let chunk = self.chunks.get_mut(&change.chunk_coords).unwrap();
            chunk.cells[change.cell_index_in_chunk] = change.new_state;
        }
    }
}
