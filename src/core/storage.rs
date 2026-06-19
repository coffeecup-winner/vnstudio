use std::collections::HashMap;

use super::types::*;

pub struct ChunkStorage<const CHUNK_SIZE: usize, State: CellState> {
    chunks: HashMap<(isize, isize), Chunk<CHUNK_SIZE, State>>,
}

impl<const CHUNK_SIZE: usize, State: CellState> ChunkStorage<CHUNK_SIZE, State> {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
        }
    }
}
