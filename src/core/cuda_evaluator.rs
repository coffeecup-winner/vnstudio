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

use std::{collections::HashMap, error::Error, io, sync::Arc, time::Duration};

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

    #[kernel]
    pub fn clear_halos<State: CellState>(mut chunks: DisjointSlice<State>) {
        let chunk_index = thread::blockIdx_x();
        let thread_index = thread::threadIdx_x();
        let chunk_start = chunk_index * (EXTENDED_CHUNK_SIZE as u32) * (EXTENDED_CHUNK_SIZE as u32);
        let default_state = State::default();

        let mut cell = thread_index;
        while cell < (EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE) as u32 {
            let x = cell % EXTENDED_CHUNK_SIZE as u32;
            let y = cell / EXTENDED_CHUNK_SIZE as u32;
            if x == 0
                || x == (EXTENDED_CHUNK_SIZE - 1) as u32
                || y == 0
                || y == (EXTENDED_CHUNK_SIZE - 1) as u32
            {
                unsafe {
                    *chunks.get_unchecked_mut((chunk_start + cell) as usize) = default_state;
                }
            }
            cell += 256;
        }
    }

    #[kernel]
    pub fn rebuild_von_neumann_halos<State: CellState>(
        mut chunks: DisjointSlice<State>,
        neighbor_indices: &[i32],
        mut missing_neighbor_flags: DisjointSlice<u8>,
    ) {
        let block = thread::blockIdx_x();
        let side = block % 4;
        let chunk_index = block / 4;
        let edge_index = thread::threadIdx_x();
        let neighbor_index = neighbor_indices[block as usize];
        let chunk_start = chunk_index * (EXTENDED_CHUNK_SIZE as u32) * (EXTENDED_CHUNK_SIZE as u32);

        let (source_index, dest_index_in_neighbor) = if side == 0 {
            (
                chunk_start + EXTENDED_CHUNK_SIZE as u32 + 1 + edge_index,
                ((EXTENDED_CHUNK_SIZE - 1) * EXTENDED_CHUNK_SIZE + 1) as u32 + edge_index,
            )
        } else if side == 1 {
            (
                chunk_start + (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32 + 1,
                (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32 + (EXTENDED_CHUNK_SIZE - 1) as u32,
            )
        } else if side == 2 {
            (
                chunk_start + (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32 + CHUNK_SIZE as u32,
                (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32,
            )
        } else {
            (
                chunk_start + (CHUNK_SIZE as u32) * EXTENDED_CHUNK_SIZE as u32 + 1 + edge_index,
                1 + edge_index,
            )
        };

        let state = unsafe { *chunks.get_unchecked_mut(source_index as usize) };
        if neighbor_index >= 0 {
            let neighbor_start =
                neighbor_index as u32 * (EXTENDED_CHUNK_SIZE as u32) * (EXTENDED_CHUNK_SIZE as u32);
            unsafe {
                *chunks.get_unchecked_mut((neighbor_start + dest_index_in_neighbor) as usize) =
                    state;
            }
        } else if state != State::default() {
            unsafe {
                *missing_neighbor_flags
                    .get_unchecked_mut((block * CHUNK_SIZE as u32 + edge_index) as usize) = 1;
            }
        }
    }
}

pub struct CudaEvaluator<State: CellState> {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: LoadedCudaModule,
    lut: Vec<State>,
    stats: CudaEvaluatorStats,
    last_halo_rebuild_on_device: bool,
}

#[derive(Default)]
struct CudaEvaluatorStats {
    total_memcopy_in: Duration,
    total_kernel_evaluate: Duration,
    total_halo_rebuild: Duration,
    total_topology_flag_copy: Duration,
    total_memcopy_out: Duration,
}

enum LoadedCudaModule {
    Embedded(kernels::LoadedModule),
    Sidecar(SidecarCudaModule),
}

struct SidecarCudaModule {
    evaluate: CudaFunction,
    clear_halos: Option<CudaFunction>,
    rebuild_von_neumann_halos: Option<CudaFunction>,
}

fn sidecar_kernel_names() -> Result<Vec<String>, Box<dyn Error>> {
    let ptx = std::fs::read_to_string("vnstudio.ptx")?;
    let entries = ptx
        .lines()
        .filter_map(|line| {
            line.trim()
                .strip_prefix(".visible .entry ")
                .and_then(|entry| entry.split('(').next())
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidData,
            "vnstudio.ptx has no entry",
        )));
    }
    Ok(entries)
}

fn sidecar_kernel_name(kernel_names: &[String], prefix: &str) -> Option<String> {
    kernel_names
        .iter()
        .find(|name| name.as_str() == prefix || name.starts_with(&format!("{prefix}_TID_")))
        .cloned()
}

fn build_neighbor_indices(chunk_coords: &[(isize, isize)]) -> Vec<i32> {
    let chunk_indices = chunk_coords
        .iter()
        .enumerate()
        .map(|(index, &coords)| (coords, index as i32))
        .collect::<HashMap<_, _>>();
    let mut neighbor_indices = Vec::with_capacity(chunk_coords.len() * 4);
    for &(x, y) in chunk_coords {
        neighbor_indices.push(*chunk_indices.get(&(x, y - 1)).unwrap_or(&-1));
        neighbor_indices.push(*chunk_indices.get(&(x - 1, y)).unwrap_or(&-1));
        neighbor_indices.push(*chunk_indices.get(&(x + 1, y)).unwrap_or(&-1));
        neighbor_indices.push(*chunk_indices.get(&(x, y + 1)).unwrap_or(&-1));
    }
    neighbor_indices
}

fn load_sidecar_module(ctx: &Arc<CudaContext>) -> Result<SidecarCudaModule, Box<dyn Error>> {
    let module = load_kernel_module(ctx, "vnstudio")?;
    let kernel_names = sidecar_kernel_names()?;
    let evaluate = sidecar_kernel_name(&kernel_names, "evaluate")
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "vnstudio.ptx has no evaluate entry",
            )
        })
        .and_then(|name| module.load_function(&name).map_err(io::Error::other))?;
    let clear_halos = sidecar_kernel_name(&kernel_names, "clear_halos")
        .and_then(|name| module.load_function(&name).ok());
    let rebuild_von_neumann_halos = sidecar_kernel_name(&kernel_names, "rebuild_von_neumann_halos")
        .and_then(|name| module.load_function(&name).ok());

    Ok(SidecarCudaModule {
        evaluate,
        clear_halos,
        rebuild_von_neumann_halos,
    })
}

impl<State: CellState, Evaluator: CellRuleEvaluator<State, VonNeumannNeighborhood<State>> + ?Sized>
    CellGridEvaluator<State, VonNeumannNeighborhood<State>, Evaluator> for CudaEvaluator<State>
where
    State: DeviceCopy,
    Chunk<State>: FillNeighborhood<State, VonNeumannNeighborhood<State>>,
{
    fn evaluate_all(
        &mut self,
        chunk_coords: &[(isize, isize)],
        input: &[Chunk<State>],
        output: &mut [Chunk<State>],
        _evaluator: &Evaluator,
    ) -> Result<(), Box<dyn Error>> {
        assert_eq!(input.len(), output.len());
        assert_eq!(input.len(), chunk_coords.len());
        self.last_halo_rebuild_on_device = false;
        if input.is_empty() {
            return Ok(());
        }

        let input_flat = flatten_chunk_cells(input);
        let output_flat = flatten_chunk_cells_mut(output);
        let neighbor_indices = build_neighbor_indices(chunk_coords);
        let missing_neighbor_flags = vec![0u8; input.len() * 4 * CHUNK_SIZE];
        let t_memcopy_in = std::time::Instant::now();
        let chunk_d = DeviceBuffer::from_host(&self.stream, input_flat)?;
        let mut chunk_new_d = DeviceBuffer::from_host(&self.stream, output_flat)?;
        let lut_d = DeviceBuffer::from_host(&self.stream, &self.lut)?;
        let neighbor_indices_d = DeviceBuffer::from_host(&self.stream, &neighbor_indices)?;
        let mut missing_neighbor_flags_d =
            DeviceBuffer::from_host(&self.stream, &missing_neighbor_flags)?;
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
                        &module.evaluate,
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

        let t_halo = std::time::Instant::now();
        let halos_rebuilt_on_device = unsafe {
            self.rebuild_halos_on_device(
                input.len(),
                &mut chunk_new_d,
                &neighbor_indices_d,
                &mut missing_neighbor_flags_d,
            )?
        };
        if halos_rebuilt_on_device {
            self.stream.synchronize()?;
            self.stats.total_halo_rebuild += t_halo.elapsed();

            let mut missing_neighbor_flags = missing_neighbor_flags;
            let t_flag = std::time::Instant::now();
            missing_neighbor_flags_d.copy_to_host(&self.stream, &mut missing_neighbor_flags)?;
            self.stats.total_topology_flag_copy += t_flag.elapsed();

            self.last_halo_rebuild_on_device =
                !missing_neighbor_flags.iter().any(|&flag| flag != 0);
        }

        let t_memcopy_out = std::time::Instant::now();
        chunk_new_d.copy_to_host(&self.stream, output_flat)?;
        self.stats.total_memcopy_out += t_memcopy_out.elapsed();

        Ok(())
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        if self.last_halo_rebuild_on_device {
            return;
        }
        rebuild_all_halos_for_storage(storage);
    }

    fn rebuild_all_halos_after_topology_change(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
        self.last_halo_rebuild_on_device = false;
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
            "Total halo rebuild: {}ms",
            self.stats.total_halo_rebuild.as_millis()
        );
        println!(
            "Total topology flag copy: {}ms",
            self.stats.total_topology_flag_copy.as_millis()
        );
        println!(
            "Total memcpy out: {}ms",
            self.stats.total_memcopy_out.as_millis()
        );
    }
}

impl<State: CellState> CudaEvaluator<State> {
    unsafe fn rebuild_halos_on_device(
        &mut self,
        chunk_count: usize,
        chunks: &mut DeviceBuffer<State>,
        neighbor_indices: &DeviceBuffer<i32>,
        missing_neighbor_flags: &mut DeviceBuffer<u8>,
    ) -> Result<bool, Box<dyn Error>>
    where
        State: DeviceCopy,
    {
        let clear_config = LaunchConfig {
            grid_dim: (chunk_count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let rebuild_config = LaunchConfig {
            grid_dim: (chunk_count as u32 * 4, 1, 1),
            block_dim: (CHUNK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        match &self.module {
            LoadedCudaModule::Embedded(module) => {
                unsafe {
                    module.clear_halos(&self.stream, clear_config, chunks)?;
                    module.rebuild_von_neumann_halos(
                        &self.stream,
                        rebuild_config,
                        chunks,
                        neighbor_indices,
                        missing_neighbor_flags,
                    )?;
                }
                Ok(true)
            }
            LoadedCudaModule::Sidecar(module) => {
                let (Some(clear_halos), Some(rebuild_von_neumann_halos)) =
                    (&module.clear_halos, &module.rebuild_von_neumann_halos)
                else {
                    return Ok(false);
                };

                let mut chunks_ptr = chunks.cu_deviceptr();
                let mut chunks_len = chunks.len() as u64;
                let mut clear_args = [
                    (&mut chunks_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut chunks_len as *mut _) as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(
                        clear_halos,
                        clear_config.grid_dim,
                        clear_config.block_dim,
                        clear_config.shared_mem_bytes,
                        &self.stream,
                        &mut clear_args,
                    )?;
                }

                let mut chunks_ptr = chunks.cu_deviceptr();
                let mut chunks_len = chunks.len() as u64;
                let mut neighbor_indices_ptr = neighbor_indices.cu_deviceptr();
                let mut neighbor_indices_len = neighbor_indices.len() as u64;
                let mut missing_neighbor_flags_ptr = missing_neighbor_flags.cu_deviceptr();
                let mut missing_neighbor_flags_len = missing_neighbor_flags.len() as u64;
                let mut rebuild_args = [
                    (&mut chunks_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut chunks_len as *mut _) as *mut std::ffi::c_void,
                    (&mut neighbor_indices_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut neighbor_indices_len as *mut _) as *mut std::ffi::c_void,
                    (&mut missing_neighbor_flags_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut missing_neighbor_flags_len as *mut _) as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(
                        rebuild_von_neumann_halos,
                        rebuild_config.grid_dim,
                        rebuild_config.block_dim,
                        rebuild_config.shared_mem_bytes,
                        &self.stream,
                        &mut rebuild_args,
                    )?;
                }

                Ok(true)
            }
        }
    }

    pub fn new(lut: Vec<State>) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let module = if let Ok(module) = load_sidecar_module(&ctx) {
            LoadedCudaModule::Sidecar(module)
        } else {
            match kernels::load(&ctx) {
                Ok(module) => LoadedCudaModule::Embedded(module),
                Err(EmbeddedModuleError::NoModules) => {
                    return Err(Box::new(io::Error::new(
                        io::ErrorKind::NotFound,
                        "no embedded CUDA module or vnstudio.ptx sidecar found",
                    )));
                }
                Err(error) => return Err(Box::new(error)),
            }
        };
        Ok(Self {
            _ctx: ctx,
            stream,
            module,
            lut,
            stats: Default::default(),
            last_halo_rebuild_on_device: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_indices_follow_top_left_right_bottom_order() {
        let coords = vec![(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)];

        assert_eq!(
            build_neighbor_indices(&coords),
            vec![
                1, 2, 3, 4, // (0, 0)
                -1, -1, -1, 0, // (0, -1)
                -1, -1, 0, -1, // (-1, 0)
                -1, 0, -1, -1, // (1, 0)
                0, -1, -1, -1, // (0, 1)
            ]
        );
    }
}
