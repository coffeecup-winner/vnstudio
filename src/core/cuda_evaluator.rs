use crate::core::{
    evaluator::rebuild_all_halos_for_storage,
    storage::{CHUNK_SIZE, Chunk, ChunkStorage, FillNeighborhood},
    types::{CellGridEvaluator, CellNeighborhood, CellRuleEvaluator, CellState},
};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

use std::error::Error;

// Size of the chunk with the external borders
const EXTENDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn evaluate(lut: &[u8], chunk: &[u8], mut chunk_new: DisjointSlice<u8>) {
        let cuda_block_x = thread::blockIdx_x();
        let cuda_block_y = thread::blockIdx_y();
        let cuda_thread_x = thread::threadIdx_x();
        let cuda_thread_y = thread::threadIdx_y();

        let chunk_index = cuda_block_x / 4;
        let chunk_idx_x = (cuda_block_x % 4) * 16 + cuda_thread_x;
        let chunk_idx_y = cuda_block_y * 16 + cuda_thread_y;

        let chunk_start_offset =
            chunk_index * (EXTENDED_CHUNK_SIZE as u32) * (EXTENDED_CHUNK_SIZE as u32);

        let cell_index = chunk_start_offset
            + (EXTENDED_CHUNK_SIZE as u32)
            + 1
            + (chunk_idx_y * EXTENDED_CHUNK_SIZE as u32)
            + chunk_idx_x;

        let left_neighbor_index = cell_index - 1;
        let right_neighbor_index = cell_index + 1;
        let top_neighbor_index = cell_index - (EXTENDED_CHUNK_SIZE as u32);
        let bottom_neighbor_index = cell_index + (EXTENDED_CHUNK_SIZE as u32);

        let cell = chunk[cell_index as usize];
        let left = chunk[left_neighbor_index as usize];
        let right = chunk[right_neighbor_index as usize];
        let top = chunk[top_neighbor_index as usize];
        let bottom = chunk[bottom_neighbor_index as usize];

        let mut lut_index = cell as usize;
        lut_index <<= 5;
        lut_index |= top as usize;
        lut_index <<= 5;
        lut_index |= left as usize;
        lut_index <<= 5;
        lut_index |= right as usize;
        lut_index <<= 5;
        lut_index |= bottom as usize;

        let new_value = lut[lut_index];
        unsafe {
            *chunk_new.get_unchecked_mut(cell_index as usize) = new_value;
        }
    }
}

pub struct CudaEvaluator;

impl<
    State: CellState,
    Neighborhood: CellNeighborhood<State>,
    Evaluator: CellRuleEvaluator<State, Neighborhood> + ?Sized,
> CellGridEvaluator<State, Neighborhood, Evaluator> for CudaEvaluator
where
    Chunk<State>: FillNeighborhood<State, Neighborhood>,
{
    fn evaluate_all(
        &mut self,
        _input: &[Chunk<State>],
        _coords: &[(isize, isize)],
        _output: &mut [Chunk<State>],
        _evaluator: &Evaluator,
    ) {
        todo!()
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
    }
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let chunk = [0u8; 200 * EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE];
    let chunk_new = [0u8; 200 * EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE];

    let lut = vec![0; 32 * 1024 * 1024];

    let chunk_d = DeviceBuffer::from_host(&stream, &chunk)?;
    let mut chunk_new_d = DeviceBuffer::from_host(&stream, &chunk_new)?;
    let lut_d = DeviceBuffer::from_host(&stream, &lut)?;

    const NUM_ITERATIONS: usize = 10000;
    let start = std::time::Instant::now();
    let module = kernels::load(&ctx)?;
    let launch_config = LaunchConfig {
        grid_dim: (200 * (CHUNK_SIZE / 16) as u32, (CHUNK_SIZE / 16) as u32, 1),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
    };
    for _ in 0..NUM_ITERATIONS {
        unsafe {
            module.evaluate(&stream, launch_config, &lut_d, &chunk_d, &mut chunk_new_d)?;
        }
        stream.synchronize()?;
    }
    let end = std::time::Instant::now();
    println!(
        "{} iterations took {}ms",
        NUM_ITERATIONS,
        (end - start).as_millis()
    );

    Ok(())
}
