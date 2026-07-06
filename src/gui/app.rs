use std::{
    collections::{BTreeSet, HashMap},
    error::Error,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    task::{Context as TaskContext, Poll},
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Color32, ColorImage, Context, PointerButton, Pos2, Rect, RichText, Sense, Stroke,
    TextureHandle, TextureOptions, Ui, Vec2,
};

use crate::{
    automata::{game_of_life::GameOfLife, von_neumann::VonNeumann},
    core::{
        cuda_evaluator::CudaEvaluator,
        golly_loader,
        storage::{Chunk, FillNeighborhood},
        types::{CellStateVisuals, CellularAutomataConfig, CellularAutomaton},
        vns_format::{
            self, LoadedVnsPattern, PatternCells, Stage, apply_game_of_life_cells,
            apply_jvn29_cells, game_of_life_cells_from_automaton, jvn29_cells_from_automaton,
        },
    },
};

use super::svg_glyph::SvgGlyph;

const INITIAL_CELL_SIZE: f32 = 28.0;
const MIN_CELL_SIZE: f32 = 1.0;
const MAX_CELL_SIZE: f32 = 160.0;
const PIXEL_RENDERING_THRESHOLD: f32 = 8.0;
const LOW_ZOOM_SCROLL_THRESHOLD: f32 = 60.0;
const FIXED_SPEED_MAX_STEPS_PER_FRAME: u32 = 8;
const RUN_TO_STAGE_MAX_STEPS_PER_FRAME: u32 = 256;
const REALTIME_FRAME_BUDGET: Duration = Duration::from_millis(12);
const REPAINT_INTERVAL: Duration = Duration::from_nanos(16_666_667);
const BASE_TITLE: &str = "VNStudio";

pub struct VnStudioApp {
    automaton: ActiveAutomaton,
    baseline_cells: PatternCells,
    zoom: f32,
    pan: Vec2,
    glyphs: HashMap<u8, SvgGlyph>,
    pixel_texture: Option<TextureHandle>,
    pixel_texture_bounds: Option<VisibleCellRange>,
    pixel_texture_generation: u64,
    simulation_generation: u64,
    pixel_zoom_scroll_accumulator: f32,
    is_running: bool,
    speed: SimulationSpeed,
    step_accumulator: f64,
    last_update_time: Option<f64>,
    ups_window_start: f64,
    updates_this_window: u32,
    last_title: Option<String>,
    breakpoints: BTreeSet<(isize, isize)>,
    last_breakpoint_hit: Option<BreakpointHit>,
    status_message: Option<String>,
    current_path: Option<PathBuf>,
    pending_file_dialog: Option<PendingFileDialog>,
    stages: Vec<Stage>,
    new_stage_name: String,
    run_to_stage: Option<RunToStage>,
}

pub enum ActiveAutomaton {
    JvN29(VonNeumann),
    GameOfLife(GameOfLife),
}

pub struct LoadedPatternForApp {
    pub automaton: ActiveAutomaton,
    pub baseline_cells: PatternCells,
    pub breakpoints: BTreeSet<(isize, isize)>,
    pub stages: Vec<Stage>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BreakpointHit {
    x: isize,
    y: isize,
    old_state: String,
    new_state: String,
    generation: u64,
}

enum PendingFileDialog {
    Load(Pin<Box<dyn Future<Output = Option<rfd::FileHandle>>>>),
    Save(Pin<Box<dyn Future<Output = Option<rfd::FileHandle>>>>),
}

enum CompletedFileDialog {
    Load(Option<PathBuf>),
    Save(Option<PathBuf>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunToStage {
    target_iteration: u64,
    ignore_breakpoints: bool,
}

impl PendingFileDialog {
    fn poll(&mut self) -> Poll<CompletedFileDialog> {
        let waker = std::task::Waker::noop();
        let mut context = TaskContext::from_waker(waker);
        match self {
            Self::Load(future) => future
                .as_mut()
                .poll(&mut context)
                .map(|handle| CompletedFileDialog::Load(handle.map(PathBuf::from))),
            Self::Save(future) => future
                .as_mut()
                .poll(&mut context)
                .map(|handle| CompletedFileDialog::Save(handle.map(PathBuf::from))),
        }
    }
}

impl ActiveAutomaton {
    pub fn new_jvn29() -> Result<Self, Box<dyn Error>> {
        if std::env::var("VNSTUDIO_CUDA").as_deref() == Ok("1") {
            Ok(Self::JvN29(VonNeumann::try_new_with_grid_evaluator(
                |lut| Ok(Box::new(CudaEvaluator::new(lut.values().to_vec())?)),
            )?))
        } else {
            Ok(Self::JvN29(VonNeumann::new()))
        }
    }

    pub fn from_loaded_vns(pattern: LoadedVnsPattern) -> Result<Self, Box<dyn Error>> {
        Self::from_pattern_cells(pattern.cells)
    }

    pub fn from_pattern_cells(cells: PatternCells) -> Result<Self, Box<dyn Error>> {
        match cells {
            PatternCells::JvN29(cells) => {
                let mut automaton = match Self::new_jvn29()? {
                    Self::JvN29(automaton) => automaton,
                    Self::GameOfLife(_) => unreachable!("new_jvn29 must return JvN29"),
                };
                apply_jvn29_cells(&cells, &mut automaton);
                Ok(Self::JvN29(automaton))
            }
            PatternCells::GameOfLife(cells) => {
                let mut automaton = GameOfLife::new();
                apply_game_of_life_cells(&cells, &mut automaton);
                Ok(Self::GameOfLife(automaton))
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::JvN29(_) => "JvN29",
            Self::GameOfLife(_) => "Game of Life",
        }
    }

    fn chunk_count(&mut self) -> usize {
        match self {
            Self::JvN29(automaton) => automaton.chunk_count(),
            Self::GameOfLife(automaton) => automaton.chunk_count(),
        }
    }

    fn evaluate_next(&mut self) {
        match self {
            Self::JvN29(automaton) => automaton.evaluate_next(),
            Self::GameOfLife(automaton) => automaton.evaluate_next(),
        }
    }

    fn get_state_display(&mut self, x: isize, y: isize) -> String {
        match self {
            Self::JvN29(automaton) => automaton.get_state(x, y).to_string(),
            Self::GameOfLife(automaton) => automaton.get_state(x, y).to_string(),
        }
    }

    fn to_pattern_cells(&mut self) -> PatternCells {
        match self {
            Self::JvN29(automaton) => PatternCells::JvN29(jvn29_cells_from_automaton(automaton)),
            Self::GameOfLife(automaton) => {
                PatternCells::GameOfLife(game_of_life_cells_from_automaton(automaton))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimulationSpeed {
    Fixed(u32),
    Realtime,
}

impl SimulationSpeed {
    fn label(self) -> &'static str {
        match self {
            SimulationSpeed::Fixed(4) => "4x",
            SimulationSpeed::Fixed(16) => "16x",
            SimulationSpeed::Fixed(64) => "64x",
            SimulationSpeed::Fixed(_) => "Custom",
            SimulationSpeed::Realtime => "Realtime",
        }
    }

    fn fixed_updates_per_second(self) -> Option<f64> {
        match self {
            SimulationSpeed::Fixed(multiplier) => Some(multiplier as f64),
            SimulationSpeed::Realtime => None,
        }
    }

    fn double(self) -> Self {
        match self {
            SimulationSpeed::Fixed(multiplier) => {
                SimulationSpeed::Fixed(multiplier.saturating_mul(2))
            }
            SimulationSpeed::Realtime => SimulationSpeed::Realtime,
        }
    }

    fn halve(self) -> Self {
        match self {
            SimulationSpeed::Fixed(multiplier) => SimulationSpeed::Fixed((multiplier / 2).max(1)),
            SimulationSpeed::Realtime => SimulationSpeed::Realtime,
        }
    }
}

impl VnStudioApp {
    pub fn new(
        _creation_context: &eframe::CreationContext<'_>,
        mut automaton: ActiveAutomaton,
        breakpoints: BTreeSet<(isize, isize)>,
        stages: Vec<Stage>,
        baseline_cells: Option<PatternCells>,
        current_path: Option<PathBuf>,
    ) -> Self {
        let baseline_cells = baseline_cells.unwrap_or_else(|| automaton.to_pattern_cells());
        Self {
            automaton,
            baseline_cells,
            zoom: INITIAL_CELL_SIZE,
            pan: Vec2::ZERO,
            glyphs: HashMap::new(),
            pixel_texture: None,
            pixel_texture_bounds: None,
            pixel_texture_generation: u64::MAX,
            simulation_generation: 0,
            pixel_zoom_scroll_accumulator: 0.0,
            is_running: false,
            speed: SimulationSpeed::Fixed(4),
            step_accumulator: 0.0,
            last_update_time: None,
            ups_window_start: 0.0,
            updates_this_window: 0,
            last_title: None,
            breakpoints,
            last_breakpoint_hit: None,
            status_message: None,
            current_path,
            pending_file_dialog: None,
            stages,
            new_stage_name: String::new(),
            run_to_stage: None,
        }
    }

    fn open_load_dialog(&mut self, ctx: &Context) {
        if self.pending_file_dialog.is_some() {
            return;
        }

        let dialog = self
            .dialog_with_current_directory(rfd::AsyncFileDialog::new())
            .add_filter("VNStudio pattern", &["vns"])
            .add_filter("Golly RLE", &["rle"]);
        self.pending_file_dialog = Some(PendingFileDialog::Load(Box::pin(dialog.pick_file())));
        self.status_message = Some("Opening load dialog...".to_string());
        ctx.request_repaint_after(REPAINT_INTERVAL);
    }

    fn finish_load_dialog(&mut self, path: PathBuf, ctx: &Context) {
        match load_pattern_from_path(&path) {
            Ok(loaded) => {
                let LoadedPatternForApp {
                    automaton,
                    baseline_cells,
                    breakpoints,
                    stages,
                } = loaded;
                self.automaton = automaton;
                self.baseline_cells = baseline_cells;
                self.breakpoints = breakpoints;
                self.stages = stages;
                self.current_path = Some(path.clone());
                self.after_pattern_replaced(ctx);
                self.status_message = Some(format!("Loaded {}", path.display()));
            }
            Err(error) => {
                self.status_message = Some(format!("Load failed: {error}"));
            }
        }
    }

    fn open_save_dialog(&mut self, ctx: &Context) {
        if self.pending_file_dialog.is_some() {
            return;
        }

        let (directory, file_name) = self.next_save_suggestion();
        let mut dialog = rfd::AsyncFileDialog::new().add_filter("VNStudio pattern", &["vns"]);
        if let Some(directory) = directory {
            dialog = dialog.set_directory(directory);
        }
        dialog = dialog.set_file_name(file_name);
        self.pending_file_dialog = Some(PendingFileDialog::Save(Box::pin(dialog.save_file())));
        self.status_message = Some("Opening save dialog...".to_string());
        ctx.request_repaint_after(REPAINT_INTERVAL);
    }

    fn finish_save_dialog(&mut self, path: PathBuf) {
        let path = with_vns_extension(path);
        match vns_format::save_vns(
            &path,
            self.baseline_cells.clone(),
            &self.breakpoints,
            &self.stages,
        ) {
            Ok(()) => {
                self.current_path = Some(path.clone());
                self.status_message = Some(format!("Saved {}", path.display()));
            }
            Err(error) => {
                self.status_message = Some(format!("Save failed: {error}"));
            }
        }
    }

    fn poll_file_dialog(&mut self, ctx: &Context) {
        let Some(pending) = &mut self.pending_file_dialog else {
            return;
        };

        match pending.poll() {
            Poll::Ready(completed) => {
                self.pending_file_dialog = None;
                match completed {
                    CompletedFileDialog::Load(Some(path)) => self.finish_load_dialog(path, ctx),
                    CompletedFileDialog::Load(None) => {
                        self.status_message = Some("Load canceled".to_string());
                    }
                    CompletedFileDialog::Save(Some(path)) => self.finish_save_dialog(path),
                    CompletedFileDialog::Save(None) => {
                        self.status_message = Some("Save canceled".to_string());
                    }
                }
            }
            Poll::Pending => {
                ctx.request_repaint_after(REPAINT_INTERVAL);
            }
        }
    }

    fn dialog_with_current_directory(&self, dialog: rfd::AsyncFileDialog) -> rfd::AsyncFileDialog {
        self.current_path
            .as_ref()
            .and_then(|path| path.parent())
            .map_or(dialog.clone(), |directory| dialog.set_directory(directory))
    }

    fn next_save_suggestion(&self) -> (Option<&Path>, String) {
        if let Some(path) = &self.current_path {
            let save_path = with_vns_extension(path.clone());
            let directory = path.parent();
            let file_name = save_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("pattern.vns")
                .to_string();
            (directory, file_name)
        } else {
            (None, "pattern.vns".to_string())
        }
    }

    fn after_pattern_replaced(&mut self, ctx: &Context) {
        self.glyphs.clear();
        self.pixel_texture = None;
        self.pixel_texture_bounds = None;
        self.pixel_texture_generation = u64::MAX;
        self.simulation_generation = 0;
        self.is_running = false;
        self.run_to_stage = None;
        self.last_breakpoint_hit = None;
        self.reset_timing(ctx.input(|input| input.time));
        self.publish_title(ctx, None);
        ctx.request_repaint();
    }

    fn draw_tools(&mut self, ui: &mut Ui) {
        ui.heading("VNStudio");
        ui.label(self.automaton.name());
        ui.separator();

        ui.horizontal(|ui| {
            let dialog_pending = self.pending_file_dialog.is_some();
            if ui
                .add_enabled(!dialog_pending, egui::Button::new("Load..."))
                .clicked()
            {
                self.open_load_dialog(ui.ctx());
            }
            if ui
                .add_enabled(!dialog_pending, egui::Button::new("Save..."))
                .clicked()
            {
                self.open_save_dialog(ui.ctx());
            }
        });
        if let Some(message) = &self.status_message {
            ui.label(message);
        }
        ui.separator();

        egui::CollapsingHeader::new("Simulation")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    let label = if self.is_running { "Pause" } else { "Run" };
                    if ui.button(label).clicked() {
                        self.is_running = !self.is_running;
                        self.reset_timing(ui.ctx().input(|input| input.time));
                    }

                    if ui.button("Step").clicked() {
                        let breakpoint_hit = self.step_simulation();
                        if breakpoint_hit {
                            self.pause_after_breakpoint(ui.ctx().input(|input| input.time));
                        }
                        self.publish_title(ui.ctx(), None);
                        if self.last_breakpoint_hit.is_none()
                            && self.is_running
                            && self.speed == SimulationSpeed::Realtime
                        {
                            self.updates_this_window += 1;
                        }
                    }

                    if ui.button("Reset to 0").clicked() {
                        match self.reset_to_baseline(ui.ctx()) {
                            Ok(()) => {
                                self.status_message = Some("Reset to iteration 0".to_string());
                            }
                            Err(error) => {
                                self.status_message = Some(format!("Reset failed: {error}"));
                            }
                        }
                    }
                });

                ui.separator();

                ui.horizontal(|ui| {
                    let fixed_speed = self.speed.fixed_updates_per_second().is_some();
                    if ui
                        .add_enabled(fixed_speed, egui::Button::new("/2"))
                        .clicked()
                    {
                        self.speed = self.speed.halve();
                        self.reset_timing(ui.ctx().input(|input| input.time));
                    }
                    if ui
                        .add_enabled(fixed_speed, egui::Button::new("*2"))
                        .clicked()
                    {
                        self.speed = self.speed.double();
                        self.reset_timing(ui.ctx().input(|input| input.time));
                    }
                    if let SimulationSpeed::Fixed(multiplier) = self.speed {
                        ui.label(format!("{multiplier}x"));
                    }
                });

                for speed in [
                    SimulationSpeed::Fixed(4),
                    SimulationSpeed::Fixed(16),
                    SimulationSpeed::Fixed(64),
                    SimulationSpeed::Realtime,
                ] {
                    if ui
                        .radio_value(&mut self.speed, speed, speed.label())
                        .changed()
                    {
                        self.reset_timing(ui.ctx().input(|input| input.time));
                    }
                }
            });

        egui::CollapsingHeader::new("Stages")
            .default_open(true)
            .show(ui, |ui| self.draw_stages(ui));

        egui::CollapsingHeader::new("Breakpoints")
            .default_open(true)
            .show(ui, |ui| {
                if let Some(hit) = &self.last_breakpoint_hit {
                    ui.label(format!(
                        "Paused at ({}, {}) on iteration {}: {} -> {}",
                        hit.x, hit.y, hit.generation, hit.old_state, hit.new_state
                    ));
                    ui.separator();
                }

                if self.breakpoints.is_empty() {
                    ui.label("No breakpoints");
                } else {
                    ui.horizontal(|ui| {
                        ui.label(format!("{} breakpoint(s)", self.breakpoints.len()));
                        if ui.button("Clear").clicked() {
                            self.breakpoints.clear();
                            self.last_breakpoint_hit = None;
                        }
                    });
                    ui.separator();

                    for &(x, y) in &self.breakpoints {
                        ui.label(format!("({}, {})", x, y));
                    }
                }
            });

        egui::CollapsingHeader::new("State Inspection")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("No cell selected");
            });
    }

    fn draw_stages(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.new_stage_name);
            if ui.button("Add Current").clicked() {
                self.add_current_stage();
            }
        });
        ui.separator();

        if self.stages.is_empty() {
            ui.label("No stages");
            return;
        }

        let mut stage_indices: Vec<_> = (0..self.stages.len()).collect();
        stage_indices.sort_by_key(|&index| {
            (
                self.stages[index].iteration,
                self.stages[index].name.to_ascii_lowercase(),
            )
        });

        let mut delete_index = None;
        for index in stage_indices {
            let stage = self.stages[index].clone();
            let passed = stage.iteration <= self.simulation_generation;
            ui.horizontal(|ui| {
                let label = format!("{} @ {}", stage.name, stage.iteration);
                if passed {
                    ui.label(RichText::new(label).weak());
                } else {
                    ui.label(label);
                    if ui.button("Play").clicked() {
                        self.start_run_to_stage(stage.iteration, false);
                    }
                    if ui.button("Skip BPs").clicked() {
                        self.start_run_to_stage(stage.iteration, true);
                    }
                }
                if ui.button("Delete").clicked() {
                    delete_index = Some(index);
                }
            });
        }

        if let Some(index) = delete_index {
            self.stages.remove(index);
            self.status_message = Some("Deleted stage".to_string());
        }
    }

    fn add_current_stage(&mut self) {
        let name = self.new_stage_name.trim();
        if name.is_empty() {
            self.status_message = Some("Stage name cannot be empty".to_string());
            return;
        }

        self.stages.push(Stage {
            name: name.to_string(),
            iteration: self.simulation_generation,
        });
        self.new_stage_name.clear();
        self.status_message = Some("Added stage".to_string());
    }

    fn start_run_to_stage(&mut self, target_iteration: u64, ignore_breakpoints: bool) {
        if target_iteration <= self.simulation_generation {
            return;
        }

        self.is_running = false;
        self.last_breakpoint_hit = None;
        self.run_to_stage = Some(RunToStage {
            target_iteration,
            ignore_breakpoints,
        });
        self.status_message = Some(format!("Running to stage at iteration {target_iteration}"));
    }

    fn reset_timing(&mut self, now: f64) {
        self.step_accumulator = 0.0;
        self.last_update_time = Some(now);
        self.ups_window_start = now;
        self.updates_this_window = 0;
    }

    fn update_simulation(&mut self, ctx: &Context) {
        let now = ctx.input(|input| input.time);
        let elapsed = self
            .last_update_time
            .replace(now)
            .map_or(0.0, |last| (now - last).max(0.0));

        if self.run_to_stage.is_some() {
            self.update_run_to_stage(now);
            ctx.request_repaint_after(REPAINT_INTERVAL);
            self.update_title(ctx, now);
            return;
        }

        if self.is_running {
            match self.speed {
                SimulationSpeed::Realtime => self.update_realtime(),
                speed => {
                    if let Some(updates_per_second) = speed.fixed_updates_per_second() {
                        let steps = fixed_steps_due(
                            &mut self.step_accumulator,
                            elapsed,
                            updates_per_second,
                            FIXED_SPEED_MAX_STEPS_PER_FRAME,
                        );
                        for _ in 0..steps {
                            if self.step_simulation() {
                                self.pause_after_breakpoint(now);
                                break;
                            }
                        }
                    }
                }
            }

            ctx.request_repaint_after(REPAINT_INTERVAL);
        }

        self.update_title(ctx, now);
    }

    fn update_run_to_stage(&mut self, now: f64) {
        let Some(run) = self.run_to_stage else {
            return;
        };

        let mut steps = 0;
        while self.simulation_generation < run.target_iteration
            && steps < RUN_TO_STAGE_MAX_STEPS_PER_FRAME
        {
            let breakpoint_hit = self.step_simulation_checked(!run.ignore_breakpoints);
            if breakpoint_hit {
                self.run_to_stage = None;
                self.pause_after_breakpoint(now);
                return;
            }
            steps += 1;
        }

        if self.simulation_generation >= run.target_iteration {
            self.run_to_stage = None;
            self.reset_timing(now);
            self.status_message = Some(format!(
                "Reached stage at iteration {}",
                run.target_iteration
            ));
        }
    }

    fn update_realtime(&mut self) {
        let start = Instant::now();
        let mut steps = 0;

        loop {
            if self.step_simulation() {
                self.pause_after_breakpoint(self.last_update_time.unwrap_or(0.0));
                break;
            }
            steps += 1;

            if start.elapsed() >= REALTIME_FRAME_BUDGET {
                break;
            }
        }

        self.updates_this_window += steps;
    }

    fn step_simulation(&mut self) -> bool {
        self.step_simulation_checked(true)
    }

    fn step_simulation_checked(&mut self, check_breakpoints: bool) -> bool {
        let breakpoint_states_before = check_breakpoints.then(|| self.breakpoint_states());
        self.automaton.evaluate_next();
        self.simulation_generation = self.simulation_generation.wrapping_add(1);
        if let Some(breakpoint_states_before) = breakpoint_states_before {
            self.last_breakpoint_hit = self
                .first_changed_breakpoint(&breakpoint_states_before, self.simulation_generation);
            self.last_breakpoint_hit.is_some()
        } else {
            false
        }
    }

    fn pause_after_breakpoint(&mut self, now: f64) {
        self.is_running = false;
        self.reset_timing(now);
    }

    fn breakpoint_states(&mut self) -> Vec<((isize, isize), String)> {
        self.breakpoints
            .iter()
            .map(|&(x, y)| ((x, y), self.automaton.get_state_display(x, y)))
            .collect()
    }

    fn first_changed_breakpoint(
        &mut self,
        states_before: &[((isize, isize), String)],
        generation: u64,
    ) -> Option<BreakpointHit> {
        states_before.iter().find_map(|((x, y), old_state)| {
            let new_state = self.automaton.get_state_display(*x, *y);
            breakpoint_state_changed(old_state, &new_state).then_some(BreakpointHit {
                x: *x,
                y: *y,
                old_state: old_state.clone(),
                new_state,
                generation,
            })
        })
    }

    fn reset_to_baseline(&mut self, ctx: &Context) -> Result<(), Box<dyn Error>> {
        self.automaton = ActiveAutomaton::from_pattern_cells(self.baseline_cells.clone())?;
        self.simulation_generation = 0;
        self.is_running = false;
        self.run_to_stage = None;
        self.last_breakpoint_hit = None;
        self.pixel_texture = None;
        self.pixel_texture_bounds = None;
        self.pixel_texture_generation = u64::MAX;
        self.reset_timing(ctx.input(|input| input.time));
        self.publish_title(ctx, None);
        ctx.request_repaint();
        Ok(())
    }

    fn update_title(&mut self, ctx: &Context, now: f64) {
        if self.is_running && self.speed == SimulationSpeed::Realtime {
            if now - self.ups_window_start >= 1.0 {
                let elapsed = (now - self.ups_window_start).max(1.0);
                let ups = (self.updates_this_window as f64 / elapsed).round() as u32;
                self.publish_title(ctx, Some(ups));
                self.ups_window_start = now;
                self.updates_this_window = 0;
            }
        } else {
            self.publish_title(ctx, None);
        }
    }

    fn publish_title(&mut self, ctx: &Context, ups: Option<u32>) {
        let ups = ups.map_or_else(|| "N/A".to_string(), |ups| ups.to_string());
        let iterations = self.simulation_generation;
        let chunks = self.automaton.chunk_count();
        let title =
            format!("{BASE_TITLE} - UPS: {ups} - Iteration: {iterations} - Chunks: {chunks}");

        if self.last_title.as_deref() != Some(&title) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.last_title = Some(title);
        }
    }

    fn draw_canvas(&mut self, ui: &mut Ui, ctx: &Context) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        painter.rect_filled(rect, 0.0, Color32::from_gray(248));

        if response.hovered() {
            let scroll_y = ctx.input(|input| input.smooth_scroll_delta.y);
            if scroll_y != 0.0 {
                let cursor = ctx.input(|input| input.pointer.hover_pos());
                if let Some(cursor) = cursor {
                    self.zoom_around(rect, cursor, scroll_y);
                }
            }
        }

        if response.dragged_by(PointerButton::Middle) {
            let delta = ctx.input(|input| input.pointer.delta());
            self.pan += delta;
            if self.zoom <= PIXEL_RENDERING_THRESHOLD {
                self.snap_pan_to_logical_pixels(rect);
            }
            ctx.request_repaint();
        }

        if response.clicked_by(PointerButton::Secondary)
            && let Some(pointer_pos) = response.interact_pointer_pos()
        {
            self.toggle_breakpoint_at(rect, pointer_pos);
        }

        self.paint_cells(rect, &painter, ctx);
        self.paint_grid(rect, &painter);
        self.paint_breakpoints(rect, &painter);
    }

    fn toggle_breakpoint_at(&mut self, rect: Rect, pointer_pos: Pos2) {
        let world = screen_to_world(rect, self.pan, self.zoom, pointer_pos);
        let cell = (world.x.floor() as isize, world.y.floor() as isize);
        if !self.breakpoints.insert(cell) {
            self.breakpoints.remove(&cell);
            if self
                .last_breakpoint_hit
                .as_ref()
                .is_some_and(|hit| (hit.x, hit.y) == cell)
            {
                self.last_breakpoint_hit = None;
            }
        }
    }

    fn zoom_around(&mut self, rect: Rect, cursor: Pos2, scroll_y: f32) {
        let before = screen_to_world(rect, self.pan, self.zoom, cursor);
        self.zoom =
            zoom_level_after_scroll(self.zoom, scroll_y, &mut self.pixel_zoom_scroll_accumulator);
        self.pan = cursor - rect.center() - before.to_vec2() * self.zoom;
        if self.zoom <= PIXEL_RENDERING_THRESHOLD {
            self.snap_pan_to_logical_pixels(rect);
        }
    }

    fn paint_cells(&mut self, rect: Rect, painter: &egui::Painter, ctx: &Context) {
        if self.zoom <= PIXEL_RENDERING_THRESHOLD {
            self.paint_pixel_cells(rect, painter, ctx);
        } else {
            self.paint_svg_cells(rect, painter);
        }
    }

    fn paint_pixel_cells(&mut self, rect: Rect, painter: &egui::Painter, ctx: &Context) {
        let visible = visible_cell_range(rect, self.pan, self.zoom);
        let texture_is_stale = self.pixel_texture_bounds.as_ref() != Some(&visible)
            || self.pixel_texture_generation != self.simulation_generation;

        if texture_is_stale {
            let image = match &mut self.automaton {
                ActiveAutomaton::JvN29(automaton) => build_pixel_image(automaton, &visible),
                ActiveAutomaton::GameOfLife(automaton) => build_pixel_image(automaton, &visible),
            };

            if let Some(texture) = &mut self.pixel_texture {
                texture.set(image, TextureOptions::NEAREST);
            } else {
                self.pixel_texture =
                    Some(ctx.load_texture("automaton-pixels", image, TextureOptions::NEAREST));
            }
            self.pixel_texture_bounds = Some(visible.clone());
            self.pixel_texture_generation = self.simulation_generation;
        }

        let Some(texture) = &self.pixel_texture else {
            return;
        };
        let destination = Rect::from_min_max(
            world_to_screen(
                rect,
                self.pan,
                self.zoom,
                visible.min_x as f32,
                visible.min_y as f32,
            ),
            world_to_screen(
                rect,
                self.pan,
                self.zoom,
                (visible.max_x + 1) as f32,
                (visible.max_y + 1) as f32,
            ),
        );
        painter.image(
            texture.id(),
            destination,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    fn paint_svg_cells(&mut self, rect: Rect, painter: &egui::Painter) {
        let visible = visible_cell_range(rect, self.pan, self.zoom);
        let padding = if self.zoom >= 12.0 { 2.0 } else { 0.5 };
        let glyphs = &mut self.glyphs;

        match &mut self.automaton {
            ActiveAutomaton::JvN29(automaton) => paint_svg_cells_for(
                automaton, glyphs, rect, self.pan, self.zoom, padding, painter, &visible,
            ),
            ActiveAutomaton::GameOfLife(automaton) => paint_svg_cells_for(
                automaton, glyphs, rect, self.pan, self.zoom, padding, painter, &visible,
            ),
        }
    }

    fn snap_pan_to_logical_pixels(&mut self, rect: Rect) {
        let origin = rect.center() + self.pan;
        let snapped_origin = Pos2::new(origin.x.round(), origin.y.round());
        self.pan = snapped_origin - rect.center();
    }

    fn paint_grid(&self, rect: Rect, painter: &egui::Painter) {
        if self.zoom <= PIXEL_RENDERING_THRESHOLD {
            return;
        }

        let visible = visible_cell_range(rect, self.pan, self.zoom);
        let stroke = if self.zoom < 14.0 {
            Stroke::new(1.0, Color32::from_gray(225))
        } else {
            Stroke::new(1.0, Color32::from_gray(205))
        };

        for x in visible.min_x..=visible.max_x + 1 {
            let screen_x = world_to_screen(rect, self.pan, self.zoom, x as f32, 0.0).x;
            painter.line_segment(
                [
                    Pos2::new(screen_x, rect.top()),
                    Pos2::new(screen_x, rect.bottom()),
                ],
                stroke,
            );
        }

        for y in visible.min_y..=visible.max_y + 1 {
            let screen_y = world_to_screen(rect, self.pan, self.zoom, 0.0, y as f32).y;
            painter.line_segment(
                [
                    Pos2::new(rect.left(), screen_y),
                    Pos2::new(rect.right(), screen_y),
                ],
                stroke,
            );
        }
    }

    fn paint_breakpoints(&self, rect: Rect, painter: &egui::Painter) {
        let visible = visible_cell_range(rect, self.pan, self.zoom);
        let stroke = Stroke::new(2.0, Color32::from_rgb(220, 20, 60));

        for &(x, y) in self.breakpoints.iter().filter(|&&(x, y)| {
            x >= visible.min_x && x <= visible.max_x && y >= visible.min_y && y <= visible.max_y
        }) {
            let cell = cell_rect(rect, self.pan, self.zoom, x, y).shrink(2.0);
            painter.rect_stroke(cell, 0.0, stroke, egui::StrokeKind::Inside);

            if self.zoom > PIXEL_RENDERING_THRESHOLD {
                let size = (self.zoom * 0.18).clamp(3.0, 8.0);
                painter.circle_filled(cell.left_top() + Vec2::splat(size), size, stroke.color);
            }
        }
    }
}

impl eframe::App for VnStudioApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.poll_file_dialog(ctx);
        self.update_simulation(ctx);

        egui::SidePanel::left("tools")
            .resizable(true)
            .default_width(260.0)
            .width_range(180.0..=420.0)
            .show(ctx, |ui| self.draw_tools(ui));

        egui::CentralPanel::default().show(ctx, |ui| self.draw_canvas(ui, ctx));
    }
}

fn paint_svg_cells_for<Config: CellularAutomataConfig>(
    automaton: &mut CellularAutomaton<Config>,
    glyphs: &mut HashMap<u8, SvgGlyph>,
    rect: Rect,
    pan: Vec2,
    zoom: f32,
    padding: f32,
    painter: &egui::Painter,
    visible: &VisibleCellRange,
) where
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    automaton.visit_non_default_cells(
        (visible.min_x, visible.min_y),
        (visible.max_x, visible.max_y),
        |x, y, state| {
            let cell_rect = cell_rect(rect, pan, zoom, x, y).shrink(padding);
            if let Some(glyph) = glyph_for(glyphs, state) {
                glyph.paint(painter, cell_rect);
            }
        },
    );
}

fn glyph_for<State: crate::core::types::CellState>(
    glyphs: &mut HashMap<u8, SvgGlyph>,
    state: State,
) -> Option<&SvgGlyph> {
    let key: u8 = state.into();
    if let std::collections::hash_map::Entry::Vacant(entry) = glyphs.entry(key) {
        let svg = state.glyph_svg()?;
        match SvgGlyph::parse(svg) {
            Ok(glyph) => {
                entry.insert(glyph);
            }
            Err(err) => {
                eprintln!("failed to parse glyph for state {state}: {err}");
                return None;
            }
        }
    }

    glyphs.get(&key)
}

pub fn load_pattern_from_path(path: &Path) -> Result<LoadedPatternForApp, Box<dyn Error>> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("vns"))
    {
        let loaded = vns_format::load_vns(path)?;
        let baseline_cells = loaded.cells.clone();
        let breakpoints = loaded.breakpoints.clone();
        let stages = loaded.stages.clone();
        return Ok(LoadedPatternForApp {
            automaton: ActiveAutomaton::from_loaded_vns(loaded)?,
            baseline_cells,
            breakpoints,
            stages,
        });
    }

    let pattern = golly_loader::load_jvn29_rle(path)?;
    let baseline_cells = PatternCells::JvN29(pattern.cells.clone());
    let mut automaton = match ActiveAutomaton::new_jvn29()? {
        ActiveAutomaton::JvN29(automaton) => automaton,
        ActiveAutomaton::GameOfLife(_) => unreachable!("new_jvn29 must return JvN29"),
    };
    pattern.apply_to(&mut automaton);
    Ok(LoadedPatternForApp {
        automaton: ActiveAutomaton::JvN29(automaton),
        baseline_cells,
        breakpoints: BTreeSet::new(),
        stages: Vec::new(),
    })
}

fn with_vns_extension(mut path: PathBuf) -> PathBuf {
    if !path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("vns"))
    {
        path.set_extension("vns");
    }
    path
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VisibleCellRange {
    min_x: isize,
    max_x: isize,
    min_y: isize,
    max_y: isize,
}

fn visible_cell_range(rect: Rect, pan: Vec2, zoom: f32) -> VisibleCellRange {
    let top_left = screen_to_world(rect, pan, zoom, rect.left_top());
    let bottom_right = screen_to_world(rect, pan, zoom, rect.right_bottom());

    VisibleCellRange {
        min_x: top_left.x.floor() as isize - 1,
        max_x: bottom_right.x.ceil() as isize + 1,
        min_y: top_left.y.floor() as isize - 1,
        max_y: bottom_right.y.ceil() as isize + 1,
    }
}

fn cell_rect(rect: Rect, pan: Vec2, zoom: f32, x: isize, y: isize) -> Rect {
    Rect::from_min_size(
        world_to_screen(rect, pan, zoom, x as f32, y as f32),
        Vec2::splat(zoom),
    )
}

fn screen_to_world(rect: Rect, pan: Vec2, zoom: f32, point: Pos2) -> Pos2 {
    let offset = point - rect.center() - pan;
    Pos2::new(offset.x / zoom, offset.y / zoom)
}

fn world_to_screen(rect: Rect, pan: Vec2, zoom: f32, x: f32, y: f32) -> Pos2 {
    rect.center() + pan + Vec2::new(x * zoom, y * zoom)
}

fn fixed_steps_due(
    accumulator: &mut f64,
    elapsed: f64,
    updates_per_second: f64,
    max_steps: u32,
) -> u32 {
    *accumulator += elapsed * updates_per_second;
    let steps = (*accumulator).floor() as u32;
    let steps = steps.min(max_steps);
    *accumulator -= steps as f64;
    steps
}

fn breakpoint_state_changed<State: Eq>(old_state: State, new_state: State) -> bool {
    old_state != new_state
}

fn build_pixel_image<Config: CellularAutomataConfig>(
    automaton: &mut CellularAutomaton<Config>,
    visible: &VisibleCellRange,
) -> ColorImage
where
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    let width = (visible.max_x - visible.min_x + 1) as usize;
    let height = (visible.max_y - visible.min_y + 1) as usize;
    let mut image = ColorImage::filled([width, height], Color32::TRANSPARENT);

    automaton.visit_non_default_cells(
        (visible.min_x, visible.min_y),
        (visible.max_x, visible.max_y),
        |x, y, state| {
            if let Some([red, green, blue]) = state.pixel_color() {
                let image_x = (x - visible.min_x) as usize;
                let image_y = (y - visible.min_y) as usize;
                image.pixels[image_y * width + image_x] = Color32::from_rgb(red, green, blue);
            }
        },
    );

    image
}

fn zoom_level_after_scroll(
    current_zoom: f32,
    scroll_y: f32,
    low_zoom_accumulator: &mut f32,
) -> f32 {
    let new_zoom = if current_zoom <= PIXEL_RENDERING_THRESHOLD {
        *low_zoom_accumulator += scroll_y;
        if low_zoom_accumulator.abs() < LOW_ZOOM_SCROLL_THRESHOLD {
            return current_zoom;
        }

        let direction = low_zoom_accumulator.signum();
        *low_zoom_accumulator -= direction * LOW_ZOOM_SCROLL_THRESHOLD;
        let integer_zoom = current_zoom.round();
        if direction > 0.0 {
            if integer_zoom >= PIXEL_RENDERING_THRESHOLD {
                PIXEL_RENDERING_THRESHOLD + 1.0
            } else {
                integer_zoom + 1.0
            }
        } else {
            integer_zoom - 1.0
        }
    } else {
        *low_zoom_accumulator = 0.0;
        let multiplier = (1.0 + scroll_y.abs() / 240.0).clamp(1.01, 1.25);
        let continuous_zoom = if scroll_y > 0.0 {
            current_zoom * multiplier
        } else {
            current_zoom / multiplier
        };

        if continuous_zoom <= PIXEL_RENDERING_THRESHOLD {
            PIXEL_RENDERING_THRESHOLD
        } else {
            continuous_zoom
        }
    };

    let clamped_zoom = new_zoom.clamp(MIN_CELL_SIZE, MAX_CELL_SIZE);
    if clamped_zoom == current_zoom {
        *low_zoom_accumulator = 0.0;
    }
    clamped_zoom
}

#[cfg(test)]
mod tests {
    use crate::automata::game_of_life::{GameOfLife, GameOfLifeState};

    use super::*;

    #[test]
    fn screen_world_round_trip() {
        let rect = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(800.0, 600.0));
        let pan = Vec2::new(40.0, -12.0);
        let zoom = 20.0;
        let screen = world_to_screen(rect, pan, zoom, -3.0, 5.0);
        let world = screen_to_world(rect, pan, zoom, screen);

        assert_eq!(world, Pos2::new(-3.0, 5.0));
    }

    #[test]
    fn fixed_steps_accumulate_fractional_time() {
        let mut accumulator = 0.0;

        assert_eq!(fixed_steps_due(&mut accumulator, 0.20, 2.0, 8), 0);
        assert_eq!(fixed_steps_due(&mut accumulator, 0.30, 2.0, 8), 1);
        assert_eq!(accumulator, 0.0);
    }

    #[test]
    fn fixed_steps_are_capped() {
        let mut accumulator = 0.0;

        assert_eq!(fixed_steps_due(&mut accumulator, 10.0, 10.0, 8), 8);
        assert!(accumulator > 0.0);
    }

    #[test]
    fn fixed_speed_buttons_step_by_two_within_bounds() {
        assert_eq!(
            SimulationSpeed::Fixed(4).double(),
            SimulationSpeed::Fixed(8)
        );
        assert_eq!(SimulationSpeed::Fixed(8).halve(), SimulationSpeed::Fixed(4));
        assert_eq!(SimulationSpeed::Fixed(4).halve(), SimulationSpeed::Fixed(2));
        assert_eq!(SimulationSpeed::Fixed(2).halve(), SimulationSpeed::Fixed(1));
        assert_eq!(
            SimulationSpeed::Fixed(64).double(),
            SimulationSpeed::Fixed(128)
        );
        assert_eq!(SimulationSpeed::Fixed(1).halve(), SimulationSpeed::Fixed(1));
        assert_eq!(
            SimulationSpeed::Realtime.double(),
            SimulationSpeed::Realtime
        );
    }

    #[test]
    fn breakpoint_state_changes_when_state_differs() {
        assert!(!breakpoint_state_changed(
            GameOfLifeState::Dead,
            GameOfLifeState::Dead
        ));
        assert!(breakpoint_state_changed(
            GameOfLifeState::Dead,
            GameOfLifeState::Live
        ));
    }

    #[test]
    fn save_paths_use_vns_extension() {
        assert_eq!(
            with_vns_extension(PathBuf::from("pattern")),
            PathBuf::from("pattern.vns")
        );
        assert_eq!(
            with_vns_extension(PathBuf::from("pattern.rle")),
            PathBuf::from("pattern.vns")
        );
        assert_eq!(
            with_vns_extension(PathBuf::from("pattern.vns")),
            PathBuf::from("pattern.vns")
        );
    }

    #[test]
    fn low_zoom_accumulates_scroll_before_stepping() {
        let mut accumulator = 0.0;

        assert_eq!(zoom_level_after_scroll(8.0, -30.0, &mut accumulator), 8.0);
        assert_eq!(accumulator, -30.0);
        assert_eq!(zoom_level_after_scroll(8.0, -30.0, &mut accumulator), 7.0);
        assert_eq!(accumulator, 0.0);
    }

    #[test]
    fn low_zoom_changes_at_most_one_level_per_call() {
        let mut accumulator = 0.0;

        assert_eq!(zoom_level_after_scroll(8.0, -150.0, &mut accumulator), 7.0);
        assert_eq!(accumulator, -90.0);
    }

    #[test]
    fn low_zoom_stays_within_bounds() {
        let mut accumulator = 0.0;

        assert_eq!(zoom_level_after_scroll(1.0, -60.0, &mut accumulator), 1.0);
        assert_eq!(accumulator, 0.0);
        assert_eq!(zoom_level_after_scroll(8.0, 60.0, &mut accumulator), 9.0);
    }

    #[test]
    fn default_state_has_no_visual() {
        let state = GameOfLifeState::default();
        assert_eq!(state.glyph_svg(), None);
        assert_eq!(state.pixel_color(), None);
    }

    #[test]
    fn pixel_image_contains_colored_live_cells() {
        let mut automaton = GameOfLife::new();
        automaton.set_state(0, 0, GameOfLifeState::Live);
        let bounds = VisibleCellRange {
            min_x: -1,
            max_x: 1,
            min_y: -1,
            max_y: 1,
        };

        let image = build_pixel_image(&mut automaton, &bounds);

        assert_eq!(image.size, [3, 3]);
        assert_eq!(image.pixels[4], Color32::from_rgb(32, 33, 36));
        assert_eq!(image.pixels[0], Color32::TRANSPARENT);
    }
}
