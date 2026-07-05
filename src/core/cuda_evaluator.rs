use crate::core::{
    evaluator::rebuild_all_halos_for_storage,
    storage::{
        CHUNK_SIZE, Chunk, ChunkStorage, FillNeighborhood, flatten_chunk_cells,
        flatten_chunk_cells_mut,
    },
    types::{CellGridEvaluator, CellRuleEvaluator, CellState, VonNeumannNeighborhood},
};

use cuda_core::{
    CudaContext, CudaFunction, CudaStream, DeviceBuffer, DeviceCopy, LaunchConfig,
    launch_kernel_on_stream,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{EmbeddedModuleError, cuda_module, load_kernel_module};

use std::{error::Error, io, sync::Arc, time::Duration};

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
    module: LoadedCudaModule,
    lut: Vec<State>,
    stats: CudaEvaluatorStats,
}

#[derive(Default)]
struct CudaEvaluatorStats {
    total_memcopy_in: Duration,
    total_kernel_evaluate: Duration,
    total_memcopy_out: Duration,
}

enum LoadedCudaModule {
    Embedded(kernels::LoadedModule),
    Sidecar(CudaFunction),
}

fn sidecar_kernel_name() -> Result<String, Box<dyn Error>> {
    let ptx = std::fs::read_to_string("vnstudio.ptx")?;
    let entry = ptx
        .lines()
        .find_map(|line| line.trim().strip_prefix(".visible .entry "))
        .and_then(|entry| entry.split('(').next())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "vnstudio.ptx has no entry"))?;
    Ok(entry.to_owned())
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
        let t_memcopy_in = std::time::Instant::now();
        let chunk_d = DeviceBuffer::from_host(&self.stream, input_flat)?;
        let mut chunk_new_d = DeviceBuffer::from_host(&self.stream, output_flat)?;
        let lut_d = DeviceBuffer::from_host(&self.stream, &self.lut)?;
        self.stats.total_memcopy_in += t_memcopy_in.elapsed();

        let launch_config = LaunchConfig {
            grid_dim: (
                input.len() as u32 * (CHUNK_SIZE / 16) as u32,
                (CHUNK_SIZE / 16) as u32,
                1,
            ),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };

        let t_kernel = std::time::Instant::now();
        unsafe {
            match &self.module {
                LoadedCudaModule::Embedded(module) => {
                    module.evaluate(
                        &self.stream,
                        launch_config,
                        &lut_d,
                        &chunk_d,
                        &mut chunk_new_d,
                    )?;
                }
                LoadedCudaModule::Sidecar(module) => {
                    let mut lut_ptr = lut_d.cu_deviceptr();
                    let mut lut_len = lut_d.len() as u64;
                    let mut chunk_ptr = chunk_d.cu_deviceptr();
                    let mut chunk_len = chunk_d.len() as u64;
                    let mut chunk_new_ptr = chunk_new_d.cu_deviceptr();
                    let mut chunk_new_len = chunk_new_d.len() as u64;
                    let mut args = [
                        (&mut lut_ptr as *mut _) as *mut std::ffi::c_void,
                        (&mut lut_len as *mut _) as *mut std::ffi::c_void,
                        (&mut chunk_ptr as *mut _) as *mut std::ffi::c_void,
                        (&mut chunk_len as *mut _) as *mut std::ffi::c_void,
                        (&mut chunk_new_ptr as *mut _) as *mut std::ffi::c_void,
                        (&mut chunk_new_len as *mut _) as *mut std::ffi::c_void,
                    ];
                    launch_kernel_on_stream(
                        module,
                        launch_config.grid_dim,
                        launch_config.block_dim,
                        launch_config.shared_mem_bytes,
                        &self.stream,
                        &mut args,
                    )?;
                }
            }
        }
        self.stream.synchronize()?;
        self.stats.total_kernel_evaluate += t_kernel.elapsed();

        let t_memcopy_out = std::time::Instant::now();
        chunk_new_d.copy_to_host(&self.stream, output_flat)?;
        self.stats.total_memcopy_out += t_memcopy_out.elapsed();

        Ok(())
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
    }

    fn print_stats(&self) {
        println!("CUDA evaluator stats:");
        println!(
            "Total memcpy in: {}ms",
            self.stats.total_memcopy_in.as_millis()
        );
        println!(
            "Total kernel evaluation: {}ms",
            self.stats.total_kernel_evaluate.as_millis()
        );
        println!(
            "Total memcpy out: {}ms",
            self.stats.total_memcopy_out.as_millis()
        );
    }
}

impl<State: CellState> CudaEvaluator<State> {
    pub fn new(lut: Vec<State>) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = match kernels::load(&ctx) {
            Ok(module) => LoadedCudaModule::Embedded(module),
            Err(EmbeddedModuleError::NoModules) => {
                let module = load_kernel_module(&ctx, "vnstudio")?;
                LoadedCudaModule::Sidecar(module.load_function(&sidecar_kernel_name()?)?)
            }
            Err(error) => return Err(Box::new(error)),
        };
        Ok(Self {
            _ctx: ctx,
            stream,
            module,
            lut,
            stats: Default::default(),
        })
    }
}
