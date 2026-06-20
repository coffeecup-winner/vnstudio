use super::{storage::*, types::*};

pub struct BasicEvaluator;

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
            let mut state = State::default();
            let mut neighborhood = Neighborhood::default();
            let mut index = chunk.get_start_index();

            for _y in 0..CHUNK_SIZE {
                for _x in 0..CHUNK_SIZE {
                    chunk.fill_neighborhood(index, &mut state, &mut neighborhood);
                    let new_state = evaluator.evaluate(state, &neighborhood);
                    if state != new_state {
                        changes.push(CellStateChange {
                            chunk_coords: *coords,
                            cell_index_in_chunk: index,
                            old_state: state,
                            new_state,
                        });
                    }
                    index += 1;
                }
                // Skip external right/left borders
                index += 2;
            }
        }

        changes
    }
}
