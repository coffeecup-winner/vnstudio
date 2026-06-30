use super::{storage::*, types::*};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

const PARALLEL_WORKER_COUNT: usize = 8;
const CHUNKS_PER_TASK: usize = 4;

#[allow(dead_code)]
pub struct BasicEvaluator;

fn evaluate_chunk<
    State: CellState,
    Neighborhood: CellNeighborhood<State>,
    Evaluator: CellRuleEvaluator<State, Neighborhood> + ?Sized,
>(
    chunk_index: usize,
    chunk: &Chunk<State>,
    evaluator: &Evaluator,
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

impl<
    State: CellState,
    Neighborhood: CellNeighborhood<State>,
    Evaluator: CellRuleEvaluator<State, Neighborhood> + ?Sized,
> CellGridEvaluator<State, Neighborhood, Evaluator> for BasicEvaluator
where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    fn evaluate_all(
        &mut self,
        storage: &ChunkStorage<State>,
        evaluator: &Evaluator,
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

pub struct ParallelEvaluator {
    pool: ThreadPool,
}

impl Default for ParallelEvaluator {
    fn default() -> Self {
        let worker_cpus = physical_worker_cpus(PARALLEL_WORKER_COUNT);
        let worker_count = worker_cpus.len().max(1);
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .start_handler(move |thread_index| {
                if let Some(&cpu) = worker_cpus.get(thread_index) {
                    pin_current_thread(cpu);
                }
            })
            .build()
            .expect("failed to create cell evaluation thread pool");

        Self { pool }
    }
}

#[cfg(target_os = "linux")]
fn physical_worker_cpus(limit: usize) -> Vec<usize> {
    use std::collections::HashSet;

    let mut physical_cores = HashSet::new();
    let mut allowed_cpus = Vec::new();
    let mut cpus = Vec::new();
    let mut allowed: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    if unsafe { libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut allowed) } != 0 {
        return (0..limit).collect();
    }

    for cpu in 0..libc::CPU_SETSIZE as usize {
        if !unsafe { libc::CPU_ISSET(cpu, &allowed) } {
            continue;
        }
        allowed_cpus.push(cpu);

        let topology = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
        let package = std::fs::read_to_string(format!("{topology}/physical_package_id"))
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok());
        let core = std::fs::read_to_string(format!("{topology}/core_id"))
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok());
        let Some(core_key) = package.zip(core) else {
            continue;
        };

        if physical_cores.insert(core_key) {
            cpus.push(cpu);
            if cpus.len() == limit {
                break;
            }
        }
    }

    if cpus.is_empty() {
        allowed_cpus.into_iter().take(limit).collect()
    } else {
        cpus
    }
}

#[cfg(not(target_os = "linux"))]
fn physical_worker_cpus(limit: usize) -> Vec<usize> {
    (0..limit.min(std::thread::available_parallelism().map_or(1, usize::from))).collect()
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) {}

impl<
    State: CellState,
    Neighborhood: CellNeighborhood<State>,
    Evaluator: CellRuleEvaluator<State, Neighborhood> + ?Sized,
> CellGridEvaluator<State, Neighborhood, Evaluator> for ParallelEvaluator
where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    fn evaluate_all(
        &mut self,
        storage: &ChunkStorage<State>,
        evaluator: &Evaluator,
    ) -> Vec<ChunkStateChanges<State>> {
        let chunks = storage.chunks();

        self.pool.install(|| {
            chunks
                .par_chunks(CHUNKS_PER_TASK)
                .enumerate()
                .flat_map_iter(|(batch_index, chunks)| {
                    let first_chunk_index = batch_index * CHUNKS_PER_TASK;
                    chunks
                        .iter()
                        .enumerate()
                        .filter_map(move |(offset, chunk)| {
                            evaluate_chunk(first_chunk_index + offset, chunk, evaluator)
                        })
                })
                .collect()
        })
    }
}
