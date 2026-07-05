use crate::core::{
    evaluator::rebuild_all_halos_for_storage, storage::{
        CHUNK_SIZE, Chunk, ChunkStorage, FillNeighborhood, flatten_chunk_cells,
        flatten_chunk_cells_mut,
    }, types::{CellGridEvaluator, CellRuleEvaluator, CellState, VonNeumannNeighborhood}
};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

use std::{error::Error, sync::Arc};

// Size of the chunk with the external borders
const EXTENDED_CHUNK_SIZE: usize = CHUNK_SIZE + 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn evaluate<State: CellState>(
        lut: &[State],
        chunk: &[State],
        mut chunk_new: DisjointSlice<State>,
    ) {
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

        let mut lut_index = Into::<u8>::into(cell) as usize;
        lut_index <<= 5;
        lut_index |= Into::<u8>::into(top) as usize;
        lut_index <<= 5;
        lut_index |= Into::<u8>::into(left) as usize;
        lut_index <<= 5;
        lut_index |= Into::<u8>::into(right) as usize;
        lut_index <<= 5;
        lut_index |= Into::<u8>::into(bottom) as usize;

        let new_value = lut[lut_index];
        unsafe {
            *chunk_new.get_unchecked_mut(cell_index as usize) = new_value;
        }
    }
}

pub struct CudaEvaluator<State: CellState> {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: kernels::LoadedModule,
    lut: Vec<State>,
}

impl<State: CellState, Evaluator: CellRuleEvaluator<State, VonNeumannNeighborhood<State>> + ?Sized>
    CellGridEvaluator<State, VonNeumannNeighborhood<State>, Evaluator> for CudaEvaluator<State>
where
    State: DeviceCopy,
    Chunk<State>: FillNeighborhood<State, VonNeumannNeighborhood<State>>,
{
    fn evaluate_all(
        &mut self,
        input: &[Chunk<State>],
        output: &mut [Chunk<State>],
        _evaluator: &Evaluator,
    ) -> Result<(), Box<dyn Error>> {
        assert_eq!(input.len(), output.len());
        if input.is_empty() {
            return Ok(());
        }

        let input_flat = flatten_chunk_cells(input);
        let output_flat = flatten_chunk_cells_mut(output);
        let chunk_d = DeviceBuffer::from_host(&self.stream, input_flat)?;
        let mut chunk_new_d = DeviceBuffer::from_host(&self.stream, output_flat)?;
        let lut_d = DeviceBuffer::from_host(&self.stream, &self.lut)?;

        let launch_config = LaunchConfig {
            grid_dim: (
                input.len() as u32 * (CHUNK_SIZE / 16) as u32,
                (CHUNK_SIZE / 16) as u32,
                1,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.module.evaluate(
                &self.stream,
                launch_config,
                &lut_d,
                &chunk_d,
                &mut chunk_new_d,
            )?;
        }
        chunk_new_d.copy_to_host(&self.stream, output_flat)?;

        Ok(())
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
    }
}

impl<State: CellState> CudaEvaluator<State> {
    pub fn new(lut: Vec<State>) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = kernels::load(&ctx)?;
        Ok(Self { _ctx: ctx, stream, module, lut })
    }
}
