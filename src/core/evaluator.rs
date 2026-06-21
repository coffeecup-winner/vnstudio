use super::{storage::*, types::*};

use rayon::prelude::*;

#[allow(dead_code)]
pub struct BasicEvaluator;

fn evaluate_chunk<State: CellState, Neighborhood: CellNeighborhood<State>>(
    chunk_index: usize,
    chunk: &Chunk<State>,
    evaluator: &dyn CellRuleEvaluator<State, Neighborhood>,
) -> Option<ChunkStateChanges<State>>
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

            // Default states can never be changed by default neighbors
            if state == State::default()
                && neighborhood
                    .neighbors()
                    .iter()
                    .all(|&n| n == State::default())
            {
                index += 1;
                continue;
            }

            let new_state = evaluator.evaluate(state, &neighborhood);
            if state != new_state {
                changes.push(CellStateChange {
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

    (!changes.is_empty()).then_some(ChunkStateChanges {
        chunk_index,
        changes,
    })
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
    ) -> Vec<ChunkStateChanges<State>> {
        let mut changes = vec![];
        for (index, chunk) in storage.chunks().iter().enumerate() {
            if let Some(chunk_changes) = evaluate_chunk(index, chunk, evaluator) {
                changes.push(chunk_changes);
            }
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
    ) -> Vec<ChunkStateChanges<State>> {
        storage
            .chunks()
            .par_iter()
            .enumerate()
            .chunks(4)
            .flat_map_iter(|chunks| {
                chunks
                    .into_iter()
                    .filter_map(|(index, chunk)| evaluate_chunk(index, chunk, evaluator))
            })
            .collect()
    }
}
