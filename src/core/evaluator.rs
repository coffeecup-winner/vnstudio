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
    chunk: &Chunk<State>,
    output: &mut Chunk<State>,
    evaluator: &Evaluator,
) where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    let mut state = State::default();
    let mut neighborhood = Neighborhood::default();
    let mut index = interior_start_index();

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
                set_interior_state(output, x, y, State::default());
                index += 1;
                continue;
            }

            let new_state = evaluator.evaluate(state, &neighborhood);
            set_interior_state(output, x, y, new_state);
            index += 1;
        }
        // Skip external right/left borders
        index += 2;
    }
}

pub(crate) fn rebuild_all_halos_for_storage<State: CellState>(storage: &mut ChunkStorage<State>) {
    for chunk in storage.chunks_mut() {
        clear_halo(chunk);
    }

    let mut updates = Vec::new();
    for (&coords, chunk) in storage.chunk_coords().iter().zip(storage.chunks()) {
        let (chunk_x, chunk_y) = coords;

        for x in 0..CHUNK_SIZE {
            let top = get_interior_state(chunk, x, 0);
            if top != State::default() {
                updates.push(BorderUpdate::Bottom {
                    coords: (chunk_x, chunk_y - 1),
                    index: x,
                    state: top,
                });
            }

            let bottom = get_interior_state(chunk, x, CHUNK_SIZE - 1);
            if bottom != State::default() {
                updates.push(BorderUpdate::Top {
                    coords: (chunk_x, chunk_y + 1),
                    index: x,
                    state: bottom,
                });
            }
        }

        for y in 0..CHUNK_SIZE {
            let left = get_interior_state(chunk, 0, y);
            if left != State::default() {
                updates.push(BorderUpdate::Right {
                    coords: (chunk_x - 1, chunk_y),
                    index: y,
                    state: left,
                });
            }

            let right = get_interior_state(chunk, CHUNK_SIZE - 1, y);
            if right != State::default() {
                updates.push(BorderUpdate::Left {
                    coords: (chunk_x + 1, chunk_y),
                    index: y,
                    state: right,
                });
            }
        }

        let top_left = get_interior_state(chunk, 0, 0);
        if top_left != State::default() {
            updates.push(BorderUpdate::BottomRight {
                coords: (chunk_x - 1, chunk_y - 1),
                state: top_left,
            });
        }

        let top_right = get_interior_state(chunk, CHUNK_SIZE - 1, 0);
        if top_right != State::default() {
            updates.push(BorderUpdate::BottomLeft {
                coords: (chunk_x + 1, chunk_y - 1),
                state: top_right,
            });
        }

        let bottom_left = get_interior_state(chunk, 0, CHUNK_SIZE - 1);
        if bottom_left != State::default() {
            updates.push(BorderUpdate::TopRight {
                coords: (chunk_x - 1, chunk_y + 1),
                state: bottom_left,
            });
        }

        let bottom_right = get_interior_state(chunk, CHUNK_SIZE - 1, CHUNK_SIZE - 1);
        if bottom_right != State::default() {
            updates.push(BorderUpdate::TopLeft {
                coords: (chunk_x + 1, chunk_y + 1),
                state: bottom_right,
            });
        }
    }

    for update in updates {
        let chunk = storage.ensure_chunk_mut(update.coords());
        update.apply(chunk);
    }
}

enum BorderUpdate<State: CellState> {
    Top {
        coords: (isize, isize),
        index: usize,
        state: State,
    },
    Bottom {
        coords: (isize, isize),
        index: usize,
        state: State,
    },
    Left {
        coords: (isize, isize),
        index: usize,
        state: State,
    },
    Right {
        coords: (isize, isize),
        index: usize,
        state: State,
    },
    TopLeft {
        coords: (isize, isize),
        state: State,
    },
    TopRight {
        coords: (isize, isize),
        state: State,
    },
    BottomLeft {
        coords: (isize, isize),
        state: State,
    },
    BottomRight {
        coords: (isize, isize),
        state: State,
    },
}

impl<State: CellState> BorderUpdate<State> {
    fn coords(&self) -> (isize, isize) {
        match *self {
            BorderUpdate::Top { coords, .. }
            | BorderUpdate::Bottom { coords, .. }
            | BorderUpdate::Left { coords, .. }
            | BorderUpdate::Right { coords, .. }
            | BorderUpdate::TopLeft { coords, .. }
            | BorderUpdate::TopRight { coords, .. }
            | BorderUpdate::BottomLeft { coords, .. }
            | BorderUpdate::BottomRight { coords, .. } => coords,
        }
    }

    fn apply(self, chunk: &mut Chunk<State>) {
        match self {
            BorderUpdate::Top { index, state, .. } => set_top_border(chunk, index, state),
            BorderUpdate::Bottom { index, state, .. } => set_bottom_border(chunk, index, state),
            BorderUpdate::Left { index, state, .. } => set_left_border(chunk, index, state),
            BorderUpdate::Right { index, state, .. } => set_right_border(chunk, index, state),
            BorderUpdate::TopLeft { state, .. } => set_top_left_corner(chunk, state),
            BorderUpdate::TopRight { state, .. } => set_top_right_corner(chunk, state),
            BorderUpdate::BottomLeft { state, .. } => set_bottom_left_corner(chunk, state),
            BorderUpdate::BottomRight { state, .. } => set_bottom_right_corner(chunk, state),
        }
    }
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
        input: &[Chunk<State>],
        output: &mut [Chunk<State>],
        evaluator: &Evaluator,
    ) {
        assert_eq!(input.len(), output.len());
        for (chunk, output) in input.iter().zip(output) {
            evaluate_chunk(chunk, output, evaluator);
        }
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
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
        input: &[Chunk<State>],
        output: &mut [Chunk<State>],
        evaluator: &Evaluator,
    ) {
        assert_eq!(input.len(), output.len());
        self.pool.install(|| {
            input
                .par_chunks(CHUNKS_PER_TASK)
                .zip(output.par_chunks_mut(CHUNKS_PER_TASK))
                .for_each(|(chunks, output_chunks)| {
                    for (chunk, output) in chunks.iter().zip(output_chunks) {
                        evaluate_chunk(chunk, output, evaluator);
                    }
                })
        });
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
    }
}
