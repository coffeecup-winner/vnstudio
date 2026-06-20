use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Color32, ColorImage, Context, PointerButton, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Ui, Vec2,
};

use crate::core::{
    storage::{Chunk, FillNeighborhood},
    types::{CellStateVisuals, CellularAutomataConfig, CellularAutomaton},
};

use super::svg_glyph::SvgGlyph;

const INITIAL_CELL_SIZE: f32 = 28.0;
const MIN_CELL_SIZE: f32 = 1.0;
const MAX_CELL_SIZE: f32 = 160.0;
const PIXEL_RENDERING_THRESHOLD: f32 = 8.0;
const LOW_ZOOM_SCROLL_THRESHOLD: f32 = 60.0;
const FIXED_SPEED_MAX_STEPS_PER_FRAME: u32 = 8;
const REALTIME_FRAME_BUDGET: Duration = Duration::from_millis(12);
const REPAINT_INTERVAL: Duration = Duration::from_nanos(16_666_667);
const BASE_TITLE: &str = "VNStudio";

pub struct VnStudioApp<Config: CellularAutomataConfig> {
    automaton: CellularAutomaton<Config>,
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimulationSpeed {
    Normal,
    Faster,
    Fast,
    Realtime,
}

impl SimulationSpeed {
    fn label(self) -> &'static str {
        match self {
            SimulationSpeed::Normal => "Normal (2 UPS)",
            SimulationSpeed::Faster => "Faster (5 UPS)",
            SimulationSpeed::Fast => "Fast (10 UPS)",
            SimulationSpeed::Realtime => "Realtime",
        }
    }

    fn fixed_updates_per_second(self) -> Option<f64> {
        match self {
            SimulationSpeed::Normal => Some(2.0),
            SimulationSpeed::Faster => Some(5.0),
            SimulationSpeed::Fast => Some(10.0),
            SimulationSpeed::Realtime => None,
        }
    }
}

impl<Config: CellularAutomataConfig> VnStudioApp<Config>
where
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    pub fn new(
        _creation_context: &eframe::CreationContext<'_>,
        automaton: CellularAutomaton<Config>,
    ) -> Self {
        Self {
            automaton,
            zoom: INITIAL_CELL_SIZE,
            pan: Vec2::ZERO,
            glyphs: HashMap::new(),
            pixel_texture: None,
            pixel_texture_bounds: None,
            pixel_texture_generation: u64::MAX,
            simulation_generation: 0,
            pixel_zoom_scroll_accumulator: 0.0,
            is_running: false,
            speed: SimulationSpeed::Normal,
            step_accumulator: 0.0,
            last_update_time: None,
            ups_window_start: 0.0,
            updates_this_window: 0,
            last_title: None,
        }
    }

    fn draw_tools(&mut self, ui: &mut Ui) {
        ui.heading("VNStudio");
        ui.separator();

        egui::CollapsingHeader::new("Breakpoints")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("No breakpoints");
            });

        egui::CollapsingHeader::new("State Inspection")
            .default_open(true)
            .show(ui, |ui| {
                ui.label("No cell selected");
            });

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
                        self.step_simulation();
                        self.publish_title(ui.ctx(), None);
                        if self.is_running && self.speed == SimulationSpeed::Realtime {
                            self.updates_this_window += 1;
                        }
                    }
                });

                ui.separator();

                for speed in [
                    SimulationSpeed::Normal,
                    SimulationSpeed::Faster,
                    SimulationSpeed::Fast,
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
                            self.step_simulation();
                        }
                    }
                }
            }

            ctx.request_repaint_after(REPAINT_INTERVAL);
        }

        self.update_title(ctx, now);
    }

    fn update_realtime(&mut self) {
        let start = Instant::now();
        let mut steps = 0;

        loop {
            self.step_simulation();
            steps += 1;

            if start.elapsed() >= REALTIME_FRAME_BUDGET {
                break;
            }
        }

        self.updates_this_window += steps;
    }

    fn step_simulation(&mut self) {
        self.automaton.evaluate_next();
        self.simulation_generation = self.simulation_generation.wrapping_add(1);
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
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::drag());
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

        self.paint_cells(rect, &painter, ctx);
        self.paint_grid(rect, &painter);
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
            let image = build_pixel_image(&self.automaton, &visible);

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
        let automaton = &self.automaton;
        let glyphs = &mut self.glyphs;

        automaton.visit_non_default_cells(
            (visible.min_x, visible.min_y),
            (visible.max_x, visible.max_y),
            |x, y, state| {
                let cell_rect = cell_rect(rect, self.pan, self.zoom, x, y).shrink(padding);
                if let Some(glyph) = Self::glyph_for(glyphs, state) {
                    glyph.paint(painter, cell_rect);
                }
            },
        );
    }

    fn snap_pan_to_logical_pixels(&mut self, rect: Rect) {
        let origin = rect.center() + self.pan;
        let snapped_origin = Pos2::new(origin.x.round(), origin.y.round());
        self.pan = snapped_origin - rect.center();
    }

    fn glyph_for(glyphs: &mut HashMap<u8, SvgGlyph>, state: Config::State) -> Option<&SvgGlyph> {
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
}

impl<Config: CellularAutomataConfig> eframe::App for VnStudioApp<Config>
where
    Chunk<Config::State>: FillNeighborhood<Config::State, Config::Neighborhood>,
{
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.update_simulation(ctx);

        egui::SidePanel::left("tools")
            .resizable(true)
            .default_width(260.0)
            .width_range(180.0..=420.0)
            .show(ctx, |ui| self.draw_tools(ui));

        egui::CentralPanel::default().show(ctx, |ui| self.draw_canvas(ui, ctx));
    }
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

fn build_pixel_image<Config: CellularAutomataConfig>(
    automaton: &CellularAutomaton<Config>,
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

        let image = build_pixel_image(&automaton, &bounds);

        assert_eq!(image.size, [3, 3]);
        assert_eq!(image.pixels[4], Color32::from_rgb(32, 33, 36));
        assert_eq!(image.pixels[0], Color32::TRANSPARENT);
    }
}
