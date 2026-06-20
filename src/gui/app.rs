use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use eframe::egui::{self, Color32, Context, PointerButton, Pos2, Rect, Sense, Stroke, Ui, Vec2};

use crate::{
    automata::game_of_life::{GameOfLife, GameOfLifeState},
    core::types::CellStateVisuals,
};

use super::svg_glyph::SvgGlyph;

const INITIAL_CELL_SIZE: f32 = 28.0;
const MIN_CELL_SIZE: f32 = 2.0;
const MAX_CELL_SIZE: f32 = 160.0;
const FIXED_SPEED_MAX_STEPS_PER_FRAME: u32 = 8;
const REALTIME_FRAME_BUDGET: Duration = Duration::from_millis(12);
const REPAINT_INTERVAL: Duration = Duration::from_nanos(16_666_667);
const BASE_TITLE: &str = "VNStudio";

pub struct VnStudioApp {
    automaton: GameOfLife,
    zoom: f32,
    pan: Vec2,
    glyphs: HashMap<u8, SvgGlyph>,
    is_running: bool,
    speed: SimulationSpeed,
    step_accumulator: f64,
    last_update_time: Option<f64>,
    ups_window_start: f64,
    updates_this_window: u32,
    title_shows_realtime_ups: bool,
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

impl VnStudioApp {
    pub fn new(_creation_context: &eframe::CreationContext<'_>) -> Self {
        let mut automaton = GameOfLife::new();
        automaton.switch_to_lut();
        seed_game_of_life(&mut automaton);

        Self {
            automaton,
            zoom: INITIAL_CELL_SIZE,
            pan: Vec2::ZERO,
            glyphs: HashMap::new(),
            is_running: false,
            speed: SimulationSpeed::Normal,
            step_accumulator: 0.0,
            last_update_time: None,
            ups_window_start: 0.0,
            updates_this_window: 0,
            title_shows_realtime_ups: false,
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
                        self.automaton.evaluate_next();
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
                            self.automaton.evaluate_next();
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
            self.automaton.evaluate_next();
            steps += 1;

            if start.elapsed() >= REALTIME_FRAME_BUDGET {
                break;
            }
        }

        self.updates_this_window += steps;
    }

    fn update_title(&mut self, ctx: &Context, now: f64) {
        if self.is_running && self.speed == SimulationSpeed::Realtime {
            if now - self.ups_window_start >= 1.0 {
                let elapsed = (now - self.ups_window_start).max(1.0);
                let ups = (self.updates_this_window as f64 / elapsed).round() as u32;
                ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
                    "{BASE_TITLE} - Realtime: {ups} UPS"
                )));
                self.ups_window_start = now;
                self.updates_this_window = 0;
                self.title_shows_realtime_ups = true;
            }
        } else if self.title_shows_realtime_ups {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(BASE_TITLE.to_string()));
            self.title_shows_realtime_ups = false;
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
            ctx.request_repaint();
        }

        self.paint_cells(rect, &painter);
        self.paint_grid(rect, &painter);
    }

    fn zoom_around(&mut self, rect: Rect, cursor: Pos2, scroll_y: f32) {
        let before = screen_to_world(rect, self.pan, self.zoom, cursor);
        let multiplier = (1.0 + scroll_y.abs() / 240.0).clamp(1.01, 1.25);
        let new_zoom = if scroll_y > 0.0 {
            self.zoom * multiplier
        } else {
            self.zoom / multiplier
        }
        .clamp(MIN_CELL_SIZE, MAX_CELL_SIZE);

        self.zoom = new_zoom;
        self.pan = cursor - rect.center() - before.to_vec2() * self.zoom;
    }

    fn paint_cells(&mut self, rect: Rect, painter: &egui::Painter) {
        let visible = visible_cell_range(rect, self.pan, self.zoom);
        let padding = if self.zoom >= 12.0 { 2.0 } else { 0.5 };

        for y in visible.min_y..=visible.max_y {
            for x in visible.min_x..=visible.max_x {
                let state = self.automaton.get_state(x, y);
                if state == GameOfLifeState::default() {
                    continue;
                }

                let cell_rect = cell_rect(rect, self.pan, self.zoom, x, y).shrink(padding);
                if let Some(glyph) = self.glyph_for(state) {
                    glyph.paint(painter, cell_rect);
                }
            }
        }
    }

    fn glyph_for(&mut self, state: GameOfLifeState) -> Option<&SvgGlyph> {
        let key = u8::from(state);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.glyphs.entry(key) {
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

        self.glyphs.get(&key)
    }

    fn paint_grid(&self, rect: Rect, painter: &egui::Painter) {
        if self.zoom < 5.0 {
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

impl eframe::App for VnStudioApp {
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

#[derive(Debug, PartialEq, Eq)]
struct VisibleCellRange {
    min_x: isize,
    max_x: isize,
    min_y: isize,
    max_y: isize,
}

fn seed_game_of_life(automaton: &mut GameOfLife) {
    automaton.set_state(1, 0, GameOfLifeState::Live);
    automaton.set_state(2, 1, GameOfLifeState::Live);
    automaton.set_state(0, 2, GameOfLifeState::Live);
    automaton.set_state(1, 2, GameOfLifeState::Live);
    automaton.set_state(2, 2, GameOfLifeState::Live);
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

#[cfg(test)]
mod tests {
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
}
