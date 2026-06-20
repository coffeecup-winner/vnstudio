use super::{storage::*, types::*};

use rayon::prelude::*;

#[allow(dead_code)]
pub struct BasicEvaluator;

fn evaluate_chunk<State: CellState, Neighborhood: CellNeighborhood<State>>(
    coords: &(isize, isize),
    chunk: &Chunk<State>,
    evaluator: &dyn CellRuleEvaluator<State, Neighborhood>,
) -> Vec<CellStateChange<State>>
where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    let mut changes = Vec::new();
    let mut state = State::default();
    let mut neighborhood = Neighborhood::default();
    let mut index = chunk.get_start_index();

    for y in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            chunk.fill_neighborhood(index, &mut state, &mut neighborhood);
            let new_state = evaluator.evaluate(state, &neighborhood);
            if state != new_state {
                changes.push(CellStateChange {
                    chunk_coords: *coords,
                    cell_index_in_chunk: (x, y),
                    old_state: state,
                    new_state,
                });
            }
            index += 1;
        }
        // Skip external right/left borders
        index += 2;
    }

    changes
}

impl<State: CellState, Neighborhood: CellNeighborhood<State>> CellGridEvaluator<State, Neighborhood>
    for BasicEvaluator
where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    fn evaluate_all(
        &mut self,
        storage: &ChunkStorage<State>,
        evaluator: &dyn CellRuleEvaluator<State, Neighborhood>,
    ) -> Vec<CellStateChange<State>> {
        let mut changes = vec![];
        for (coords, chunk) in storage.all_chunks_iter() {
            changes.extend(evaluate_chunk(coords, chunk, evaluator));
        }

        changes
    }
}

pub struct ParallelEvaluator;

impl<State: CellState, Neighborhood: CellNeighborhood<State>> CellGridEvaluator<State, Neighborhood>
    for ParallelEvaluator
where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    fn evaluate_all(
        &mut self,
        storage: &ChunkStorage<State>,
        evaluator: &dyn CellRuleEvaluator<State, Neighborhood>,
    ) -> Vec<CellStateChange<State>> {
        storage
            .all_chunks_iter()
            .par_bridge()
            .flat_map_iter(|(coords, chunk)| evaluate_chunk(coords, chunk, evaluator))
            .collect()
    }
}
