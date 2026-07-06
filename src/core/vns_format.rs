use std::{
    collections::BTreeSet,
    error::Error,
    fmt::{Display, Formatter},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    automata::{
        game_of_life::{GameOfLife, GameOfLifeState},
        von_neumann::{VonNeumann, VonNeumannState},
    },
    core::{golly_loader, types::Cell},
};

const FORMAT_NAME: &str = "vnstudio.pattern";
const FORMAT_VERSION: u32 = 1;
const RLE_LINE_LENGTH: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ruleset {
    JvN29,
    GameOfLife,
}

impl Ruleset {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::JvN29 => "jvn29",
            Self::GameOfLife => "game_of_life",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternCells {
    JvN29(Vec<Cell<VonNeumannState>>),
    GameOfLife(Vec<Cell<GameOfLifeState>>),
}

impl PatternCells {
    pub fn ruleset(&self) -> Ruleset {
        match self {
            Self::JvN29(_) => Ruleset::JvN29,
            Self::GameOfLife(_) => Ruleset::GameOfLife,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedVnsPattern {
    pub cells: PatternCells,
    pub breakpoints: BTreeSet<(isize, isize)>,
    pub stages: Vec<Stage>,
    pub overlays: Vec<Overlay>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stage {
    pub name: String,
    pub iteration: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Overlay {
    #[serde(rename = "directed_lines")]
    DirectedLines(DirectedLineOverlay),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectedLineOverlay {
    pub name: String,
    pub visible: bool,
    pub lines: Vec<DirectedLine>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectedLine {
    pub name: String,
    pub visible: bool,
    pub points: Vec<Coordinate>,
}

#[derive(Debug)]
pub enum VnsError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Parse(String),
}

impl VnsError {
    fn parse(message: impl Into<String>) -> Self {
        Self::Parse(message.into())
    }
}

impl Display for VnsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Parse(message) => formatter.write_str(message),
        }
    }
}

impl Error for VnsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Parse(_) => None,
        }
    }
}

impl From<std::io::Error> for VnsError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for VnsError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct VnsDocument {
    format: String,
    version: u32,
    ruleset: String,
    pattern: RlePattern,
    #[serde(default)]
    extra: ExtraState,
}

#[derive(Debug, Serialize, Deserialize)]
struct RlePattern {
    encoding: String,
    origin: Coordinate,
    size: PatternSize,
    lines: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coordinate {
    pub x: isize,
    pub y: isize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct PatternSize {
    width: usize,
    height: usize,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ExtraState {
    #[serde(default)]
    breakpoints: Vec<Coordinate>,
    #[serde(default)]
    stages: Vec<Stage>,
    #[serde(default)]
    overlays: Vec<Overlay>,
}

pub fn load_vns(path: impl AsRef<Path>) -> Result<LoadedVnsPattern, VnsError> {
    parse_vns(&fs::read_to_string(path)?)
}

pub fn save_vns(
    path: impl AsRef<Path>,
    cells: PatternCells,
    breakpoints: &BTreeSet<(isize, isize)>,
    stages: &[Stage],
    overlays: &[Overlay],
) -> Result<(), VnsError> {
    fs::write(path, serialize_vns(cells, breakpoints, stages, overlays)?)?;
    Ok(())
}

pub fn parse_vns(input: &str) -> Result<LoadedVnsPattern, VnsError> {
    let document: VnsDocument = serde_json::from_str(input)?;
    if document.format != FORMAT_NAME {
        return Err(VnsError::parse(format!(
            "unsupported format {:?}; expected {FORMAT_NAME:?}",
            document.format
        )));
    }
    if document.version != FORMAT_VERSION {
        return Err(VnsError::parse(format!(
            "unsupported .vns version {}; expected {FORMAT_VERSION}",
            document.version
        )));
    }
    if document.pattern.encoding != "rle" {
        return Err(VnsError::parse(format!(
            "unsupported pattern encoding {:?}; expected \"rle\"",
            document.pattern.encoding
        )));
    }

    let breakpoints = document
        .extra
        .breakpoints
        .into_iter()
        .map(|coordinate| (coordinate.x, coordinate.y))
        .collect();
    let stages = document.extra.stages;
    let overlays = document.extra.overlays;
    validate_overlays(&overlays)?;

    let cells = match document.ruleset.as_str() {
        "jvn29" => PatternCells::JvN29(parse_jvn29_rle_pattern(&document.pattern)?),
        "game_of_life" => {
            PatternCells::GameOfLife(parse_game_of_life_rle_pattern(&document.pattern)?)
        }
        ruleset => {
            return Err(VnsError::parse(format!(
                "unsupported ruleset {ruleset:?}; expected \"jvn29\" or \"game_of_life\""
            )));
        }
    };

    Ok(LoadedVnsPattern {
        cells,
        breakpoints,
        stages,
        overlays,
    })
}

pub fn serialize_vns(
    cells: PatternCells,
    breakpoints: &BTreeSet<(isize, isize)>,
    stages: &[Stage],
    overlays: &[Overlay],
) -> Result<String, VnsError> {
    validate_overlays(overlays)?;
    let ruleset = cells.ruleset();
    let pattern = match cells {
        PatternCells::JvN29(cells) => build_rle_pattern(&cells, jvn29_state_token)?,
        PatternCells::GameOfLife(cells) => build_rle_pattern(&cells, game_of_life_state_token)?,
    };
    let document = VnsDocument {
        format: FORMAT_NAME.to_string(),
        version: FORMAT_VERSION,
        ruleset: ruleset.as_str().to_string(),
        pattern,
        extra: ExtraState {
            breakpoints: breakpoints
                .iter()
                .map(|&(x, y)| Coordinate { x, y })
                .collect(),
            stages: stages.to_vec(),
            overlays: overlays.to_vec(),
        },
    };

    Ok(serde_json::to_string_pretty(&document)?)
}

fn validate_overlays(overlays: &[Overlay]) -> Result<(), VnsError> {
    for overlay in overlays {
        match overlay {
            Overlay::DirectedLines(overlay) => {
                for line in &overlay.lines {
                    validate_directed_line(line)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_directed_line(line: &DirectedLine) -> Result<(), VnsError> {
    if line.points.len() < 2 {
        return Err(VnsError::parse(format!(
            "directed line {:?} must have at least two points",
            line.name
        )));
    }

    for pair in line.points.windows(2) {
        if !coordinates_are_axis_aligned(pair[0], pair[1]) {
            return Err(VnsError::parse(format!(
                "directed line {:?} contains a non-orthogonal segment",
                line.name
            )));
        }
        if pair[0] == pair[1] {
            return Err(VnsError::parse(format!(
                "directed line {:?} contains a zero-length segment",
                line.name
            )));
        }
    }

    Ok(())
}

pub fn coordinates_are_axis_aligned(start: Coordinate, end: Coordinate) -> bool {
    start.x == end.x || start.y == end.y
}

pub fn jvn29_cells_from_automaton(automaton: &mut VonNeumann) -> Vec<Cell<VonNeumannState>> {
    let mut cells = Vec::new();
    automaton.visit_all_non_default_cells(|x, y, state| cells.push(Cell { x, y, state }));
    cells
}

pub fn game_of_life_cells_from_automaton(automaton: &mut GameOfLife) -> Vec<Cell<GameOfLifeState>> {
    let mut cells = Vec::new();
    automaton.visit_all_non_default_cells(|x, y, state| cells.push(Cell { x, y, state }));
    cells
}

pub fn apply_jvn29_cells(cells: &[Cell<VonNeumannState>], automaton: &mut VonNeumann) {
    for cell in cells {
        automaton.set_state(cell.x, cell.y, cell.state);
    }
}

pub fn apply_game_of_life_cells(cells: &[Cell<GameOfLifeState>], automaton: &mut GameOfLife) {
    for cell in cells {
        automaton.set_state(cell.x, cell.y, cell.state);
    }
}

fn parse_jvn29_rle_pattern(pattern: &RlePattern) -> Result<Vec<Cell<VonNeumannState>>, VnsError> {
    let input = format!(
        "#CXRLE Pos={},{}\nx = {}, y = {}, rule = JvN29\n{}",
        pattern.origin.x,
        pattern.origin.y,
        pattern.size.width,
        pattern.size.height,
        pattern.lines.concat()
    );
    let pattern = golly_loader::parse_jvn29_rle(&input)
        .map_err(|error| VnsError::parse(format!("invalid JvN29 RLE: {error}")))?;
    Ok(pattern.cells)
}

fn parse_game_of_life_rle_pattern(
    pattern: &RlePattern,
) -> Result<Vec<Cell<GameOfLifeState>>, VnsError> {
    parse_rle_data(
        &pattern.lines.concat(),
        pattern.origin,
        |token| match token {
            "o" => Ok(GameOfLifeState::Live),
            "b" | "." => Ok(GameOfLifeState::Dead),
            token => Err(VnsError::parse(format!(
                "invalid Game of Life RLE token {token:?}"
            ))),
        },
    )
}

fn parse_rle_data<State: Copy + Default + PartialEq>(
    data: &str,
    origin: Coordinate,
    parse_state: impl Fn(&str) -> Result<State, VnsError>,
) -> Result<Vec<Cell<State>>, VnsError>
where
    State: crate::core::types::CellState,
{
    let bytes = data.as_bytes();
    let mut cells = Vec::new();
    let mut x = 0isize;
    let mut y = 0isize;
    let mut count = 0usize;
    let mut index = 0;
    let mut ended = false;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            if count != 0 {
                return Err(VnsError::parse(
                    "whitespace is not allowed after a run count",
                ));
            }
            index += 1;
            continue;
        }
        if byte.is_ascii_digit() {
            if byte == b'0' && count == 0 {
                return Err(VnsError::parse("run counts cannot have a leading zero"));
            }
            count = count
                .checked_mul(10)
                .and_then(|value| value.checked_add((byte - b'0') as usize))
                .ok_or_else(|| VnsError::parse("run count overflow"))?;
            index += 1;
            continue;
        }

        let run = if count == 0 { 1 } else { count };
        count = 0;
        match byte {
            b'$' => {
                x = 0;
                y = checked_add_coord(y, run)?;
            }
            b'!' => {
                if run != 1 {
                    return Err(VnsError::parse("the ! terminator cannot have a run count"));
                }
                ended = true;
                index += 1;
                break;
            }
            b'b' | b'.' | b'o' => {
                let token = std::str::from_utf8(&bytes[index..index + 1])
                    .expect("single ASCII RLE token must be valid UTF-8");
                let state = parse_state(token)?;
                add_state_run(&mut cells, &mut x, y, run, state, origin)?;
            }
            _ => {
                return Err(VnsError::parse(format!(
                    "invalid RLE token {:?}",
                    byte as char
                )));
            }
        }
        index += 1;
    }

    if count != 0 {
        return Err(VnsError::parse("run count is missing a value"));
    }
    if ended
        && bytes[index..]
            .iter()
            .any(|byte| !byte.is_ascii_whitespace())
    {
        return Err(VnsError::parse("unexpected data after ! terminator"));
    }

    Ok(cells)
}

fn add_state_run<State: crate::core::types::CellState>(
    cells: &mut Vec<Cell<State>>,
    x: &mut isize,
    y: isize,
    run: usize,
    state: State,
    origin: Coordinate,
) -> Result<(), VnsError> {
    let world_y = origin
        .y
        .checked_add(y)
        .ok_or_else(|| VnsError::parse("y coordinate overflow"))?;

    for offset in 0..run {
        let local_x = checked_add_coord(*x, offset)?;
        let world_x = origin
            .x
            .checked_add(local_x)
            .ok_or_else(|| VnsError::parse("x coordinate overflow"))?;
        if state != State::default() {
            cells.push(Cell {
                x: world_x,
                y: world_y,
                state,
            });
        }
    }
    *x = checked_add_coord(*x, run)?;
    Ok(())
}

fn checked_add_coord(coordinate: isize, amount: usize) -> Result<isize, VnsError> {
    let amount = isize::try_from(amount).map_err(|_| VnsError::parse("coordinate overflow"))?;
    coordinate
        .checked_add(amount)
        .ok_or_else(|| VnsError::parse("coordinate overflow"))
}

fn build_rle_pattern<State: crate::core::types::CellState>(
    cells: &[Cell<State>],
    state_token: impl Fn(State) -> Result<&'static str, VnsError>,
) -> Result<RlePattern, VnsError> {
    let Some(bounds) = bounds(cells) else {
        return Ok(RlePattern {
            encoding: "rle".to_string(),
            origin: Coordinate { x: 0, y: 0 },
            size: PatternSize {
                width: 0,
                height: 0,
            },
            lines: vec!["!".to_string()],
        });
    };

    let width = usize::try_from(bounds.max_x - bounds.min_x + 1)
        .map_err(|_| VnsError::parse("pattern width overflow"))?;
    let height = usize::try_from(bounds.max_y - bounds.min_y + 1)
        .map_err(|_| VnsError::parse("pattern height overflow"))?;
    let mut sorted = cells.to_vec();
    sorted.sort_by_key(|cell| (cell.y, cell.x));

    let mut output = String::new();
    let mut cell_index = 0usize;
    for y in bounds.min_y..=bounds.max_y {
        let mut pending_token: Option<&'static str> = None;
        let mut pending_run = 0usize;

        for x in bounds.min_x..=bounds.max_x {
            let state = if cell_index < sorted.len()
                && sorted[cell_index].x == x
                && sorted[cell_index].y == y
            {
                let state = sorted[cell_index].state;
                cell_index += 1;
                state
            } else {
                State::default()
            };
            let token = state_token(state)?;
            push_rle_run(&mut output, &mut pending_token, &mut pending_run, token);
        }
        flush_rle_run(&mut output, &mut pending_token, &mut pending_run);
        if y != bounds.max_y {
            output.push('$');
        }
    }
    output.push('!');

    Ok(RlePattern {
        encoding: "rle".to_string(),
        origin: Coordinate {
            x: bounds.min_x,
            y: bounds.min_y,
        },
        size: PatternSize { width, height },
        lines: split_rle_lines(&output),
    })
}

#[derive(Clone, Copy)]
struct Bounds {
    min_x: isize,
    max_x: isize,
    min_y: isize,
    max_y: isize,
}

fn bounds<State: crate::core::types::CellState>(cells: &[Cell<State>]) -> Option<Bounds> {
    let first = cells.first()?;
    let mut bounds = Bounds {
        min_x: first.x,
        max_x: first.x,
        min_y: first.y,
        max_y: first.y,
    };

    for cell in &cells[1..] {
        bounds.min_x = bounds.min_x.min(cell.x);
        bounds.max_x = bounds.max_x.max(cell.x);
        bounds.min_y = bounds.min_y.min(cell.y);
        bounds.max_y = bounds.max_y.max(cell.y);
    }

    Some(bounds)
}

fn push_rle_run(
    output: &mut String,
    pending_token: &mut Option<&'static str>,
    pending_run: &mut usize,
    token: &'static str,
) {
    if *pending_token == Some(token) {
        *pending_run += 1;
    } else {
        flush_rle_run(output, pending_token, pending_run);
        *pending_token = Some(token);
        *pending_run = 1;
    }
}

fn flush_rle_run(
    output: &mut String,
    pending_token: &mut Option<&'static str>,
    pending_run: &mut usize,
) {
    let Some(token) = pending_token.take() else {
        return;
    };
    if *pending_run > 1 {
        output.push_str(&pending_run.to_string());
    }
    output.push_str(token);
    *pending_run = 0;
}

fn split_rle_lines(input: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();

    for character in input.chars() {
        current.push(character);
        if current.len() >= RLE_LINE_LENGTH || character == '$' || character == '!' {
            lines.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }

    lines
}

fn game_of_life_state_token(state: GameOfLifeState) -> Result<&'static str, VnsError> {
    Ok(match state {
        GameOfLifeState::Dead => "b",
        GameOfLifeState::Live => "o",
    })
}

fn jvn29_state_token(state: VonNeumannState) -> Result<&'static str, VnsError> {
    use VonNeumannState::*;

    Ok(match state {
        Ground => ".",
        Sensitized => "A",
        Sensitized0 => "B",
        Sensitized1 => "C",
        Sensitized00 => "D",
        Sensitized01 => "E",
        Sensitized10 => "F",
        Sensitized11 => "G",
        Sensitized000 => "H",
        TransmissionQuiescentRight => "I",
        TransmissionQuiescentUp => "J",
        TransmissionQuiescentLeft => "K",
        TransmissionQuiescentDown => "L",
        TransmissionExcitedRight => "M",
        TransmissionExcitedUp => "N",
        TransmissionExcitedLeft => "O",
        TransmissionExcitedDown => "P",
        SpecialTransmissionQuiescentRight => "Q",
        SpecialTransmissionQuiescentUp => "R",
        SpecialTransmissionQuiescentLeft => "S",
        SpecialTransmissionQuiescentDown => "T",
        SpecialTransmissionExcitedRight => "U",
        SpecialTransmissionExcitedUp => "V",
        SpecialTransmissionExcitedLeft => "W",
        SpecialTransmissionExcitedDown => "X",
        Confluent00 => "pA",
        Confluent10 => "pB",
        Confluent01 => "pC",
        Confluent11 => "pD",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn game_of_life_round_trips_rle_and_breakpoints() {
        let cells = vec![
            Cell {
                x: -1,
                y: 2,
                state: GameOfLifeState::Live,
            },
            Cell {
                x: 0,
                y: 2,
                state: GameOfLifeState::Live,
            },
            Cell {
                x: 1,
                y: 2,
                state: GameOfLifeState::Live,
            },
        ];
        let breakpoints = BTreeSet::from([(0, 2)]);
        let stages = vec![Stage {
            name: "Blinker".to_string(),
            iteration: 1,
        }];
        let overlays = vec![Overlay::DirectedLines(DirectedLineOverlay {
            name: "Wires".to_string(),
            visible: true,
            lines: vec![DirectedLine {
                name: "Line 1".to_string(),
                visible: false,
                points: vec![
                    Coordinate { x: -1, y: 2 },
                    Coordinate { x: 2, y: 2 },
                    Coordinate { x: 2, y: 4 },
                ],
            }],
        })];

        let serialized = serialize_vns(
            PatternCells::GameOfLife(cells.clone()),
            &breakpoints,
            &stages,
            &overlays,
        )
        .unwrap();
        let loaded = parse_vns(&serialized).unwrap();

        assert_eq!(loaded.cells, PatternCells::GameOfLife(cells));
        assert_eq!(loaded.breakpoints, breakpoints);
        assert_eq!(loaded.stages, stages);
        assert_eq!(loaded.overlays, overlays);
        assert!(serialized.contains("\"encoding\": \"rle\""));
        assert!(serialized.contains("\"stages\""));
        assert!(serialized.contains("\"kind\": \"directed_lines\""));
        assert!(serialized.contains("3o!"));
    }

    #[test]
    fn jvn29_round_trips_golly_style_tokens() {
        let cells = vec![
            Cell {
                x: 5,
                y: -3,
                state: VonNeumannState::Sensitized,
            },
            Cell {
                x: 6,
                y: -3,
                state: VonNeumannState::Confluent11,
            },
        ];

        let serialized = serialize_vns(
            PatternCells::JvN29(cells.clone()),
            &BTreeSet::new(),
            &[],
            &[],
        )
        .unwrap();
        let loaded = parse_vns(&serialized).unwrap();

        assert_eq!(loaded.cells, PatternCells::JvN29(cells));
        assert!(serialized.contains("ApD!"));
    }

    #[test]
    fn missing_extra_items_default_to_empty() {
        let input = r#"{
            "format": "vnstudio.pattern",
            "version": 1,
            "ruleset": "game_of_life",
            "pattern": {
                "encoding": "rle",
                "origin": { "x": 0, "y": 0 },
                "size": { "width": 1, "height": 1 },
                "lines": ["o!"]
            },
            "extra": {
                "breakpoints": []
            }
        }"#;

        let loaded = parse_vns(input).unwrap();

        assert!(loaded.stages.is_empty());
        assert!(loaded.overlays.is_empty());
    }

    #[test]
    fn rejects_non_orthogonal_directed_line_segments() {
        let input = r#"{
            "format": "vnstudio.pattern",
            "version": 1,
            "ruleset": "game_of_life",
            "pattern": {
                "encoding": "rle",
                "origin": { "x": 0, "y": 0 },
                "size": { "width": 1, "height": 1 },
                "lines": ["o!"]
            },
            "extra": {
                "overlays": [{
                    "kind": "directed_lines",
                    "name": "Bad",
                    "visible": true,
                    "lines": [{
                        "name": "Diagonal",
                        "visible": true,
                        "points": [
                            { "x": 0, "y": 0 },
                            { "x": 1, "y": 1 }
                        ]
                    }]
                }]
            }
        }"#;

        assert!(parse_vns(input).is_err());
    }

    #[test]
    fn rejects_unknown_ruleset() {
        let input = r#"{
            "format": "vnstudio.pattern",
            "version": 1,
            "ruleset": "unknown",
            "pattern": {
                "encoding": "rle",
                "origin": { "x": 0, "y": 0 },
                "size": { "width": 0, "height": 0 },
                "lines": ["!"]
            }
        }"#;

        assert!(parse_vns(input).is_err());
    }

    #[test]
    fn rejects_unsupported_version() {
        let input = r#"{
            "format": "vnstudio.pattern",
            "version": 999,
            "ruleset": "game_of_life",
            "pattern": {
                "encoding": "rle",
                "origin": { "x": 0, "y": 0 },
                "size": { "width": 0, "height": 0 },
                "lines": ["!"]
            }
        }"#;

        assert!(parse_vns(input).is_err());
    }

    #[test]
    fn rejects_malformed_game_of_life_rle() {
        let input = r#"{
            "format": "vnstudio.pattern",
            "version": 1,
            "ruleset": "game_of_life",
            "pattern": {
                "encoding": "rle",
                "origin": { "x": 0, "y": 0 },
                "size": { "width": 1, "height": 1 },
                "lines": ["A!"]
            }
        }"#;

        assert!(parse_vns(input).is_err());
    }
}
