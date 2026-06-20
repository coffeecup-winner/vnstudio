use std::{
    error::Error,
    fmt::{Display, Formatter},
    fs,
    path::Path,
};

use crate::{
    automata::von_neumann::{VonNeumann, VonNeumannState},
    core::types::Cell,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GollyJvN29Pattern {
    pub declared_size: (usize, usize),
    pub origin: (isize, isize),
    pub generation: Option<String>,
    pub comments: Vec<String>,
    pub cells: Vec<Cell<VonNeumannState>>,
}

impl GollyJvN29Pattern {
    pub fn apply_to(&self, automaton: &mut VonNeumann) {
        for cell in &self.cells {
            automaton.set_state(cell.x, cell.y, cell.state);
        }
    }
}

#[derive(Debug)]
pub enum GollyLoadError {
    Io(std::io::Error),
    Parse { line: usize, message: String },
}

impl GollyLoadError {
    fn parse(line: usize, message: impl Into<String>) -> Self {
        Self::Parse {
            line,
            message: message.into(),
        }
    }
}

impl Display for GollyLoadError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GollyLoadError::Io(error) => write!(formatter, "{error}"),
            GollyLoadError::Parse { line, message } => {
                write!(formatter, "line {line}: {message}")
            }
        }
    }
}

impl Error for GollyLoadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            GollyLoadError::Io(error) => Some(error),
            GollyLoadError::Parse { .. } => None,
        }
    }
}

impl From<std::io::Error> for GollyLoadError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub fn load_jvn29_rle(path: impl AsRef<Path>) -> Result<GollyJvN29Pattern, GollyLoadError> {
    parse_jvn29_rle(&fs::read_to_string(path)?)
}

pub fn parse_jvn29_rle(input: &str) -> Result<GollyJvN29Pattern, GollyLoadError> {
    let mut declared_size = None;
    let mut origin = (0, 0);
    let mut generation = None;
    let mut comments = Vec::new();
    let mut declared_rule = None;
    let mut data = String::new();
    let mut first_data_line = 1;

    for (line_index, raw_line) in input.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("#CXRLE") {
            parse_xrle_line(line, line_number, &mut origin, &mut generation)?;
        } else if line.starts_with('#') {
            if let Some(rule) = parse_legacy_rule_line(line) {
                set_rule(&mut declared_rule, rule, line_number)?;
            } else {
                comments.push(line.to_string());
            }
        } else if is_header_line(line) {
            let header = parse_header(line, line_number)?;
            declared_size = Some(header.size);
            if let Some(rule) = header.rule {
                set_rule(&mut declared_rule, &rule, line_number)?;
            }
        } else {
            if data.is_empty() {
                first_data_line = line_number;
            }
            data.push_str(line);
        }
    }

    let declared_size =
        declared_size.ok_or_else(|| GollyLoadError::parse(1, "missing RLE x/y header"))?;
    let rule =
        declared_rule.ok_or_else(|| GollyLoadError::parse(1, "missing JvN29 rule declaration"))?;
    validate_rule(&rule, 1)?;
    let cells = parse_data(&data, first_data_line, origin)?;

    Ok(GollyJvN29Pattern {
        declared_size,
        origin,
        generation,
        comments,
        cells,
    })
}

struct Header {
    size: (usize, usize),
    rule: Option<String>,
}

fn is_header_line(line: &str) -> bool {
    let mut chars = line.chars();
    chars.next().is_some_and(|character| character == 'x')
        && chars
            .next()
            .is_some_and(|character| character.is_ascii_whitespace() || character == '=')
}

fn parse_header(line: &str, line_number: usize) -> Result<Header, GollyLoadError> {
    let mut width = None;
    let mut height = None;
    let mut rule = None;

    for part in line.split(',') {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| GollyLoadError::parse(line_number, "malformed header field"))?;
        match key.trim().to_ascii_lowercase().as_str() {
            "x" => {
                width = Some(parse_usize(value.trim(), line_number, "invalid width")?);
            }
            "y" => {
                height = Some(parse_usize(value.trim(), line_number, "invalid height")?);
            }
            "rule" => rule = Some(value.trim().to_string()),
            field => {
                return Err(GollyLoadError::parse(
                    line_number,
                    format!("unsupported header field {field:?}"),
                ));
            }
        }
    }

    Ok(Header {
        size: (
            width.ok_or_else(|| GollyLoadError::parse(line_number, "missing x dimension"))?,
            height.ok_or_else(|| GollyLoadError::parse(line_number, "missing y dimension"))?,
        ),
        rule,
    })
}

fn parse_legacy_rule_line(line: &str) -> Option<&str> {
    let rest = line
        .strip_prefix("#r")
        .or_else(|| line.strip_prefix("#R"))?;
    let rule = rest.trim();
    (!rule.is_empty()).then_some(rule)
}

fn set_rule(
    declared_rule: &mut Option<String>,
    rule: &str,
    line_number: usize,
) -> Result<(), GollyLoadError> {
    validate_rule(rule, line_number)?;
    if let Some(previous) = declared_rule {
        if !rules_equal(previous, rule) {
            return Err(GollyLoadError::parse(
                line_number,
                "conflicting rule declarations",
            ));
        }
    } else {
        *declared_rule = Some(rule.to_string());
    }
    Ok(())
}

fn rules_equal(left: &str, right: &str) -> bool {
    normalize_rule(left) == normalize_rule(right)
}

fn normalize_rule(rule: &str) -> String {
    rule.trim().to_ascii_lowercase()
}

fn validate_rule(rule: &str, line_number: usize) -> Result<(), GollyLoadError> {
    let normalized = normalize_rule(rule);
    if normalized.contains(':') {
        return Err(GollyLoadError::parse(
            line_number,
            "bounded JvN29 rules are not supported",
        ));
    }
    if normalized != "jvn29" && normalized != "jvn-29" {
        return Err(GollyLoadError::parse(
            line_number,
            format!("unsupported rule {rule:?}; expected JvN29"),
        ));
    }
    Ok(())
}

fn parse_xrle_line(
    line: &str,
    line_number: usize,
    origin: &mut (isize, isize),
    generation: &mut Option<String>,
) -> Result<(), GollyLoadError> {
    for field in line["#CXRLE".len()..].split_whitespace() {
        let (key, value) = field
            .split_once('=')
            .ok_or_else(|| GollyLoadError::parse(line_number, "malformed CXRLE field"))?;
        match key.to_ascii_lowercase().as_str() {
            "pos" => {
                let (x, y) = value
                    .split_once(',')
                    .ok_or_else(|| GollyLoadError::parse(line_number, "malformed CXRLE Pos"))?;
                *origin = (
                    parse_isize(x, line_number, "invalid CXRLE x position")?,
                    parse_isize(y, line_number, "invalid CXRLE y position")?,
                );
            }
            "gen" => {
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err(GollyLoadError::parse(
                        line_number,
                        "invalid CXRLE generation",
                    ));
                }
                *generation = Some(value.to_string());
            }
            _ => {}
        }
    }
    Ok(())
}

fn parse_data(
    data: &str,
    line_number: usize,
    origin: (isize, isize),
) -> Result<Vec<Cell<VonNeumannState>>, GollyLoadError> {
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
                return Err(GollyLoadError::parse(
                    line_number,
                    "whitespace is not allowed after a run count",
                ));
            }
            index += 1;
            continue;
        }
        if byte.is_ascii_digit() {
            if byte == b'0' && count == 0 {
                return Err(GollyLoadError::parse(
                    line_number,
                    "run counts cannot have a leading zero",
                ));
            }
            count = count
                .checked_mul(10)
                .and_then(|value| value.checked_add((byte - b'0') as usize))
                .ok_or_else(|| GollyLoadError::parse(line_number, "run count overflow"))?;
            index += 1;
            continue;
        }

        let run = if count == 0 { 1 } else { count };
        count = 0;
        match byte {
            b'.' | b'b' => x = checked_add_coord(x, run, line_number)?,
            b'$' => {
                x = 0;
                y = checked_add_coord(y, run, line_number)?;
            }
            b'!' => {
                if run != 1 {
                    return Err(GollyLoadError::parse(
                        line_number,
                        "the ! terminator cannot have a run count",
                    ));
                }
                ended = true;
                index += 1;
                break;
            }
            b'o' => {
                add_state_run(&mut cells, &mut x, y, run, 1, origin, line_number)?;
            }
            b'A'..=b'X' => {
                let state = byte - b'A' + 1;
                add_state_run(&mut cells, &mut x, y, run, state, origin, line_number)?;
            }
            b'p' => {
                let suffix = *bytes.get(index + 1).ok_or_else(|| {
                    GollyLoadError::parse(line_number, "incomplete p-state token")
                })?;
                if !(b'A'..=b'D').contains(&suffix) {
                    return Err(GollyLoadError::parse(
                        line_number,
                        "JvN29 only permits pA through pD",
                    ));
                }
                let state = 25 + suffix - b'A';
                add_state_run(&mut cells, &mut x, y, run, state, origin, line_number)?;
                index += 1;
            }
            _ => {
                return Err(GollyLoadError::parse(
                    line_number,
                    format!("invalid RLE token {:?}", byte as char),
                ));
            }
        }
        index += 1;
    }

    if count != 0 {
        return Err(GollyLoadError::parse(
            line_number,
            "run count is missing a value",
        ));
    }
    if ended
        && bytes[index..]
            .iter()
            .any(|byte| !byte.is_ascii_whitespace())
    {
        return Err(GollyLoadError::parse(
            line_number,
            "unexpected data after ! terminator",
        ));
    }

    Ok(cells)
}

fn add_state_run(
    cells: &mut Vec<Cell<VonNeumannState>>,
    x: &mut isize,
    y: isize,
    run: usize,
    golly_state: u8,
    origin: (isize, isize),
    line_number: usize,
) -> Result<(), GollyLoadError> {
    let state = map_golly_state(golly_state)
        .ok_or_else(|| GollyLoadError::parse(line_number, "JvN29 state is out of range"))?;
    let world_y = origin
        .1
        .checked_add(y)
        .ok_or_else(|| GollyLoadError::parse(line_number, "y coordinate overflow"))?;

    for offset in 0..run {
        let local_x = checked_add_coord(*x, offset, line_number)?;
        let world_x = origin
            .0
            .checked_add(local_x)
            .ok_or_else(|| GollyLoadError::parse(line_number, "x coordinate overflow"))?;
        cells.push(Cell {
            x: world_x,
            y: world_y,
            state,
        });
    }
    *x = checked_add_coord(*x, run, line_number)?;
    Ok(())
}

fn checked_add_coord(
    coordinate: isize,
    amount: usize,
    line_number: usize,
) -> Result<isize, GollyLoadError> {
    let amount = isize::try_from(amount)
        .map_err(|_| GollyLoadError::parse(line_number, "coordinate overflow"))?;
    coordinate
        .checked_add(amount)
        .ok_or_else(|| GollyLoadError::parse(line_number, "coordinate overflow"))
}

fn parse_usize(value: &str, line_number: usize, message: &str) -> Result<usize, GollyLoadError> {
    value
        .parse()
        .map_err(|_| GollyLoadError::parse(line_number, message))
}

fn parse_isize(value: &str, line_number: usize, message: &str) -> Result<isize, GollyLoadError> {
    value
        .parse()
        .map_err(|_| GollyLoadError::parse(line_number, message))
}

fn map_golly_state(state: u8) -> Option<VonNeumannState> {
    use VonNeumannState::*;

    Some(match state {
        0 => Ground,
        1 => Sensitized,
        2 => Sensitized0,
        3 => Sensitized1,
        4 => Sensitized00,
        5 => Sensitized01,
        6 => Sensitized10,
        7 => Sensitized11,
        8 => Sensitized000,
        9 => TransmissionQuiescentRight,
        10 => TransmissionQuiescentUp,
        11 => TransmissionQuiescentLeft,
        12 => TransmissionQuiescentDown,
        13 => TransmissionExcitedRight,
        14 => TransmissionExcitedUp,
        15 => TransmissionExcitedLeft,
        16 => TransmissionExcitedDown,
        17 => SpecialTransmissionQuiescentRight,
        18 => SpecialTransmissionQuiescentUp,
        19 => SpecialTransmissionQuiescentLeft,
        20 => SpecialTransmissionQuiescentDown,
        21 => SpecialTransmissionExcitedRight,
        22 => SpecialTransmissionExcitedUp,
        23 => SpecialTransmissionExcitedLeft,
        24 => SpecialTransmissionExcitedDown,
        25 => Confluent00,
        26 => Confluent10,
        27 => Confluent01,
        28 => Confluent11,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_jvn29_states() {
        let pattern =
            parse_jvn29_rle("x = 28, y = 1, rule = JvN29\nABCDEFGHIJKLMNOPQRSTUVWXpApBpCpD!")
                .unwrap();

        assert_eq!(pattern.cells.len(), 28);
        for (index, cell) in pattern.cells.iter().enumerate() {
            assert_eq!(cell.state, map_golly_state(index as u8 + 1).unwrap());
            assert_eq!((cell.x, cell.y), (index as isize, 0));
        }
    }

    #[test]
    fn parses_runs_rows_and_xrle_metadata() {
        let pattern = parse_jvn29_rle(
            "#CXRLE Pos=-5,7 Gen=3480106827776\n\
             # test comment\n\
             x = 5, y = 3, rule = JvN-29\n\
             2A2. B$2$3pD!",
        )
        .unwrap();

        assert_eq!(pattern.declared_size, (5, 3));
        assert_eq!(pattern.origin, (-5, 7));
        assert_eq!(pattern.generation.as_deref(), Some("3480106827776"));
        assert_eq!(pattern.comments, vec!["# test comment"]);
        assert_eq!(
            pattern
                .cells
                .iter()
                .map(|cell| (cell.x, cell.y, cell.state))
                .collect::<Vec<_>>(),
            vec![
                (-5, 7, VonNeumannState::Sensitized),
                (-4, 7, VonNeumannState::Sensitized),
                (-1, 7, VonNeumannState::Sensitized0),
                (-5, 10, VonNeumannState::Confluent11),
                (-4, 10, VonNeumannState::Confluent11),
                (-3, 10, VonNeumannState::Confluent11),
            ]
        );
    }

    #[test]
    fn accepts_legacy_rule_declaration() {
        let pattern = parse_jvn29_rle("#r jVn29\nx = 1, y = 1\nA").unwrap();
        assert_eq!(pattern.cells.len(), 1);
    }

    #[test]
    fn rejects_other_rules_and_out_of_range_states() {
        assert!(parse_jvn29_rle("x = 1, y = 1, rule = B3/S23\nA!").is_err());
        assert!(parse_jvn29_rle("x = 1, y = 1, rule = Nobili32\nA!").is_err());
        assert!(parse_jvn29_rle("x = 1, y = 1, rule = JvN29:T10,10\nA!").is_err());
        assert!(parse_jvn29_rle("x = 1, y = 1, rule = JvN29\npE!").is_err());
    }

    #[test]
    fn applies_pattern_to_automaton() {
        let pattern = parse_jvn29_rle("#CXRLE Pos=-2,3\nx = 1, y = 1, rule = JvN29\nA!").unwrap();
        let mut automaton = VonNeumann::new();

        pattern.apply_to(&mut automaton);

        assert_eq!(automaton.get_state(-2, 3), VonNeumannState::Sensitized);
    }

    #[test]
    fn parses_real_golly_jvn29_fragment() {
        let pattern = parse_jvn29_rle(
            "x = 60, y = 2, rule = JvN29\n\
             5.2pA.pA.pA3.pA.2pA2.pA3.3pA4.2pA2.2pA2.pA2.pA.pA2.pA.3pA.3pA.2pA$\n\
             4.pA3.pA.2pA.2pA.pA.pA.pA3.pA5.pA3.pA2.pA.pA2.pA.2pA.pA2.pA2.pA3.pA.pA!",
        )
        .unwrap();

        assert_eq!(pattern.cells.len(), 49);
        assert!(
            pattern
                .cells
                .iter()
                .all(|cell| cell.state == VonNeumannState::Confluent00)
        );
    }
}
