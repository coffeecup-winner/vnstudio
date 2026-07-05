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
    pub fn clear_topology_flag(mut topology_changed: DisjointSlice<u8>) {
        unsafe {
            *topology_changed.get_unchecked_mut(0) = 0;
        }
    }

    #[kernel]
    pub fn apply_von_neumann_halo_ops<State: CellState>(
        mut chunks: DisjointSlice<State>,
        halo_ops: &[u32],
        mut topology_changed: DisjointSlice<u8>,
    ) {
        let op_index = block_linear_index();
        let op_offset = op_index * 3;
        if op_offset + 2 >= halo_ops.len() as u32 {
            return;
        }

        let source_index = halo_ops[op_offset as usize];
        let dest_index = halo_ops[(op_offset + 1) as usize];
        let flags = halo_ops[(op_offset + 2) as usize];

        let state = unsafe { *chunks.get_unchecked_mut(source_index as usize) };
        if flags == 0 {
            unsafe {
                *chunks.get_unchecked_mut(dest_index as usize) = state;
            }
        } else {
            let default_state = State::default();
            unsafe {
                *chunks.get_unchecked_mut(dest_index as usize) = default_state;
                if state != default_state {
                    *topology_changed.get_unchecked_mut(0) = 1;
                }
            }
        }
    }

    fn block_linear_index() -> u32 {
        thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()
    }
}

pub struct CudaEvaluator<State: CellState> {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: LoadedCudaModule,
    lut_d: DeviceBuffer<State>,
    device_current: Option<DeviceBuffer<State>>,
    device_next: Option<DeviceBuffer<State>>,
    halo_ops_d: Option<DeviceBuffer<u32>>,
    topology_changed_d: DeviceBuffer<u8>,
    device_topology: Vec<(isize, isize)>,
    device_chunk_count: usize,
    host_synced: bool,
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
    fast_path_iterations: u64,
    fallback_iterations: u64,
}

enum LoadedCudaModule {
    Embedded(kernels::LoadedModule),
    Sidecar(SidecarCudaModule),
}

struct SidecarCudaModule {
    evaluate: CudaFunction,
    clear_topology_flag: Option<CudaFunction>,
    apply_von_neumann_halo_ops: Option<CudaFunction>,
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

const HALO_OP_STRIDE: usize = 3;
const HALO_OP_MISSING_NEIGHBOR: u32 = 1;

fn build_halo_ops(chunk_coords: &[(isize, isize)]) -> Vec<u32> {
    let chunk_indices = chunk_coords
        .iter()
        .enumerate()
        .map(|(index, &coords)| (coords, index as i32))
        .collect::<HashMap<_, _>>();
    let mut halo_ops = Vec::with_capacity(chunk_coords.len() * 4 * CHUNK_SIZE * HALO_OP_STRIDE);
    for (chunk_index, &(x, y)) in chunk_coords.iter().enumerate() {
        let chunk_start = (chunk_index * EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE) as u32;
        let neighbors = [
            chunk_indices.get(&(x, y - 1)).copied(),
            chunk_indices.get(&(x - 1, y)).copied(),
            chunk_indices.get(&(x + 1, y)).copied(),
            chunk_indices.get(&(x, y + 1)).copied(),
        ];

        for edge_index in 0..CHUNK_SIZE as u32 {
            for (side, neighbor_index) in neighbors.iter().enumerate() {
                let (source_index, dest_index_in_neighbor, missing_dest_index) = match side {
                    0 => (
                        chunk_start + EXTENDED_CHUNK_SIZE as u32 + 1 + edge_index,
                        ((EXTENDED_CHUNK_SIZE - 1) * EXTENDED_CHUNK_SIZE + 1) as u32 + edge_index,
                        chunk_start + 1 + edge_index,
                    ),
                    1 => (
                        chunk_start + (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32 + 1,
                        (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32
                            + (EXTENDED_CHUNK_SIZE - 1) as u32,
                        chunk_start + (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32,
                    ),
                    2 => (
                        chunk_start
                            + (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32
                            + CHUNK_SIZE as u32,
                        (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32,
                        chunk_start
                            + (edge_index + 1) * EXTENDED_CHUNK_SIZE as u32
                            + (EXTENDED_CHUNK_SIZE - 1) as u32,
                    ),
                    _ => (
                        chunk_start
                            + (CHUNK_SIZE as u32) * EXTENDED_CHUNK_SIZE as u32
                            + 1
                            + edge_index,
                        1 + edge_index,
                        chunk_start
                            + ((EXTENDED_CHUNK_SIZE - 1) * EXTENDED_CHUNK_SIZE) as u32
                            + 1
                            + edge_index,
                    ),
                };

                if let Some(neighbor_index) = neighbor_index {
                    let neighbor_start = *neighbor_index as u32
                        * (EXTENDED_CHUNK_SIZE as u32)
                        * (EXTENDED_CHUNK_SIZE as u32);
                    halo_ops.push(source_index);
                    halo_ops.push(neighbor_start + dest_index_in_neighbor);
                    halo_ops.push(0);
                } else {
                    halo_ops.push(source_index);
                    halo_ops.push(missing_dest_index);
                    halo_ops.push(HALO_OP_MISSING_NEIGHBOR);
                }
            }
        }
    }
    halo_ops
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
    let clear_topology_flag = sidecar_kernel_name(&kernel_names, "clear_topology_flag")
        .and_then(|name| module.load_function(&name).ok());
    let apply_von_neumann_halo_ops =
        sidecar_kernel_name(&kernel_names, "apply_von_neumann_halo_ops")
            .and_then(|name| module.load_function(&name).ok());

    Ok(SidecarCudaModule {
        evaluate,
        clear_topology_flag,
        apply_von_neumann_halo_ops,
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

        self.ensure_device_state(chunk_coords, input, output)?;

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
        let chunk_d = self
            .device_current
            .as_ref()
            .expect("device current buffer must be initialized");
        let chunk_new_d = self
            .device_next
            .as_mut()
            .expect("device next buffer must be initialized");
        unsafe {
            match &self.module {
                LoadedCudaModule::Embedded(module) => {
                    module.evaluate(
                        &self.stream,
                        launch_config,
                        &self.lut_d,
                        chunk_d,
                        chunk_new_d,
                    )?;
                }
                LoadedCudaModule::Sidecar(module) => {
                    let mut lut_ptr = self.lut_d.cu_deviceptr();
                    let mut lut_len = self.lut_d.len() as u64;
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
        let mut chunk_new_d = self
            .device_next
            .take()
            .expect("device next buffer must be initialized");
        let halo_ops_d = self
            .halo_ops_d
            .take()
            .expect("halo ops buffer must be initialized");
        let halos_rebuilt_on_device = unsafe {
            Self::rebuild_halos_on_device(
                &self.module,
                &self.stream,
                self.device_chunk_count,
                &mut chunk_new_d,
                &halo_ops_d,
                &mut self.topology_changed_d,
            )?
        };
        if halos_rebuilt_on_device {
            self.stream.synchronize()?;
            self.stats.total_halo_rebuild += t_halo.elapsed();

            let mut topology_changed = [0u8];
            let t_flag = std::time::Instant::now();
            self.topology_changed_d
                .copy_to_host(&self.stream, &mut topology_changed)?;
            self.stats.total_topology_flag_copy += t_flag.elapsed();

            self.last_halo_rebuild_on_device = topology_changed[0] == 0;
        }

        self.halo_ops_d = Some(halo_ops_d);

        if self.last_halo_rebuild_on_device {
            let old_current = self
                .device_current
                .replace(chunk_new_d)
                .expect("device current buffer must be initialized");
            self.device_next = Some(old_current);
            self.host_synced = false;
            self.stats.fast_path_iterations += 1;
        } else {
            let output_flat = flatten_chunk_cells_mut(output);
            let t_memcopy_out = std::time::Instant::now();
            chunk_new_d.copy_to_host(&self.stream, output_flat)?;
            self.stats.total_memcopy_out += t_memcopy_out.elapsed();
            self.device_next = Some(chunk_new_d);
            self.invalidate_device_state();
            self.host_synced = true;
            self.stats.fallback_iterations += 1;
        }

        Ok(())
    }

    fn rebuild_all_halos(&mut self, storage: &mut ChunkStorage<State>) {
        if self.last_halo_rebuild_on_device {
            return;
        }
        rebuild_all_halos_for_storage(storage);
        self.invalidate_device_state();
    }

    fn rebuild_all_halos_after_topology_change(&mut self, storage: &mut ChunkStorage<State>) {
        rebuild_all_halos_for_storage(storage);
        self.last_halo_rebuild_on_device = false;
        self.invalidate_device_state();
    }

    fn sync_to_host_if_needed(
        &mut self,
        storage: &mut ChunkStorage<State>,
    ) -> Result<(), Box<dyn Error>> {
        if self.host_synced {
            return Ok(());
        }

        let Some(device_current) = &self.device_current else {
            self.host_synced = true;
            return Ok(());
        };

        let t_memcopy_out = std::time::Instant::now();
        device_current.copy_to_host(&self.stream, storage.active_cells_flat_mut())?;
        self.stats.total_memcopy_out += t_memcopy_out.elapsed();
        self.host_synced = true;

        Ok(())
    }

    fn storage_changed(&mut self) {
        self.invalidate_device_state();
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
        println!("Fast path iterations: {}", self.stats.fast_path_iterations);
        println!("Fallback iterations: {}", self.stats.fallback_iterations);
    }
}

impl<State: CellState + DeviceCopy> CudaEvaluator<State> {
    fn invalidate_device_state(&mut self) {
        self.device_current = None;
        self.device_next = None;
        self.halo_ops_d = None;
        self.device_topology.clear();
        self.device_chunk_count = 0;
        self.host_synced = true;
        self.last_halo_rebuild_on_device = false;
    }

    fn ensure_device_state(
        &mut self,
        chunk_coords: &[(isize, isize)],
        input: &[Chunk<State>],
        output: &[Chunk<State>],
    ) -> Result<(), Box<dyn Error>>
    where
        State: DeviceCopy,
    {
        let needs_upload = self.device_current.is_none()
            || self.device_next.is_none()
            || self.halo_ops_d.is_none()
            || self.device_chunk_count != input.len()
            || self.device_topology.as_slice() != chunk_coords;

        if !needs_upload {
            return Ok(());
        }

        let t_memcopy_in = std::time::Instant::now();
        let input_flat = flatten_chunk_cells(input);
        let output_flat = flatten_chunk_cells(output);
        let halo_ops = build_halo_ops(chunk_coords);

        self.device_current = Some(DeviceBuffer::from_host(&self.stream, input_flat)?);
        self.device_next = Some(DeviceBuffer::from_host(&self.stream, output_flat)?);
        self.halo_ops_d = Some(DeviceBuffer::from_host(&self.stream, &halo_ops)?);
        self.device_topology.clear();
        self.device_topology.extend_from_slice(chunk_coords);
        self.device_chunk_count = input.len();
        self.host_synced = true;
        self.stats.total_memcopy_in += t_memcopy_in.elapsed();

        Ok(())
    }

    unsafe fn rebuild_halos_on_device(
        module: &LoadedCudaModule,
        stream: &Arc<CudaStream>,
        chunk_count: usize,
        chunks: &mut DeviceBuffer<State>,
        halo_ops: &DeviceBuffer<u32>,
        topology_changed: &mut DeviceBuffer<u8>,
    ) -> Result<bool, Box<dyn Error>>
    where
        State: DeviceCopy,
    {
        let rebuild_config = LaunchConfig {
            grid_dim: ((chunk_count * 4 * CHUNK_SIZE).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let flag_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        match module {
            LoadedCudaModule::Embedded(module) => {
                unsafe {
                    module.clear_topology_flag(stream, flag_config, topology_changed)?;
                    module.apply_von_neumann_halo_ops(
                        stream,
                        rebuild_config,
                        chunks,
                        halo_ops,
                        topology_changed,
                    )?;
                }
                Ok(true)
            }
            LoadedCudaModule::Sidecar(module) => {
                let (Some(clear_topology_flag), Some(apply_von_neumann_halo_ops)) = (
                    &module.clear_topology_flag,
                    &module.apply_von_neumann_halo_ops,
                ) else {
                    return Ok(false);
                };

                let mut topology_changed_ptr = topology_changed.cu_deviceptr();
                let mut topology_changed_len = topology_changed.len() as u64;
                let mut flag_args = [
                    (&mut topology_changed_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut topology_changed_len as *mut _) as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(
                        clear_topology_flag,
                        flag_config.grid_dim,
                        flag_config.block_dim,
                        flag_config.shared_mem_bytes,
                        stream,
                        &mut flag_args,
                    )?;
                }

                let mut chunks_ptr = chunks.cu_deviceptr();
                let mut chunks_len = chunks.len() as u64;
                let mut halo_ops_ptr = halo_ops.cu_deviceptr();
                let mut halo_ops_len = halo_ops.len() as u64;
                let mut topology_changed_ptr = topology_changed.cu_deviceptr();
                let mut topology_changed_len = topology_changed.len() as u64;
                let mut rebuild_args = [
                    (&mut chunks_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut chunks_len as *mut _) as *mut std::ffi::c_void,
                    (&mut halo_ops_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut halo_ops_len as *mut _) as *mut std::ffi::c_void,
                    (&mut topology_changed_ptr as *mut _) as *mut std::ffi::c_void,
                    (&mut topology_changed_len as *mut _) as *mut std::ffi::c_void,
                ];
                unsafe {
                    launch_kernel_on_stream(
                        apply_von_neumann_halo_ops,
                        rebuild_config.grid_dim,
                        rebuild_config.block_dim,
                        rebuild_config.shared_mem_bytes,
                        stream,
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
        let lut_d = DeviceBuffer::from_host(&stream, &lut)?;
        let topology_changed_d = DeviceBuffer::from_host(&stream, &[0u8])?;
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
            lut_d,
            device_current: None,
            device_next: None,
            halo_ops_d: None,
            topology_changed_d,
            device_topology: Vec::new(),
            device_chunk_count: 0,
            host_synced: true,
            stats: Default::default(),
            last_halo_rebuild_on_device: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halo_ops_copy_to_existing_cardinal_neighbors() {
        let coords = vec![(0, 0), (0, -1), (-1, 0), (1, 0), (0, 1)];
        let ops = build_halo_ops(&coords);
        let chunk_cells = (EXTENDED_CHUNK_SIZE * EXTENDED_CHUNK_SIZE) as u32;

        assert_eq!(ops.len(), coords.len() * 4 * CHUNK_SIZE * HALO_OP_STRIDE);
        assert_eq!(
            &ops[0..12],
            vec![
                // top edge x=0 -> bottom halo of top neighbor
                EXTENDED_CHUNK_SIZE as u32 + 1,
                chunk_cells + ((EXTENDED_CHUNK_SIZE - 1) * EXTENDED_CHUNK_SIZE + 1) as u32,
                0,
                // left edge y=0 -> right halo of left neighbor
                EXTENDED_CHUNK_SIZE as u32 + 1,
                chunk_cells * 2 + (EXTENDED_CHUNK_SIZE * 1 + EXTENDED_CHUNK_SIZE - 1) as u32,
                0,
                // right edge y=0 -> left halo of right neighbor
                EXTENDED_CHUNK_SIZE as u32 + CHUNK_SIZE as u32,
                chunk_cells * 3 + EXTENDED_CHUNK_SIZE as u32,
                0,
                // bottom edge x=0 -> top halo of bottom neighbor
                (CHUNK_SIZE * EXTENDED_CHUNK_SIZE + 1) as u32,
                chunk_cells * 4 + 1,
                0,
            ]
        );
    }

    #[test]
    fn halo_ops_clear_missing_cardinal_neighbors() {
        let ops = build_halo_ops(&[(0, 0)]);

        assert_eq!(ops.len(), 4 * CHUNK_SIZE * HALO_OP_STRIDE);
        assert_eq!(
            &ops[0..12],
            vec![
                EXTENDED_CHUNK_SIZE as u32 + 1,
                1,
                HALO_OP_MISSING_NEIGHBOR,
                EXTENDED_CHUNK_SIZE as u32 + 1,
                EXTENDED_CHUNK_SIZE as u32,
                HALO_OP_MISSING_NEIGHBOR,
                EXTENDED_CHUNK_SIZE as u32 + CHUNK_SIZE as u32,
                (EXTENDED_CHUNK_SIZE * 2 - 1) as u32,
                HALO_OP_MISSING_NEIGHBOR,
                (CHUNK_SIZE * EXTENDED_CHUNK_SIZE + 1) as u32,
                ((EXTENDED_CHUNK_SIZE - 1) * EXTENDED_CHUNK_SIZE + 1) as u32,
                HALO_OP_MISSING_NEIGHBOR,
            ]
        );
    }
}
