use std::fmt::Display;

use crate::core::types::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use strum::EnumCount;

#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, IntoPrimitive, TryFromPrimitive, EnumCount,
)]
#[repr(u8)]
pub enum VonNeumannState {
    // Default empty state
    #[default]
    Ground,

    // Transition/sensitized states
    Sensitized,
    Sensitized0,
    Sensitized00,
    Sensitized000,
    Sensitized01,
    Sensitized1,
    Sensitized10,
    Sensitized11,

    // Confluent states
    Confluent00,
    Confluent01,
    Confluent10,
    Confluent11,

    // Ordinary transmission states
    TransmissionExcitedUp,
    TransmissionExcitedDown,
    TransmissionExcitedLeft,
    TransmissionExcitedRight,
    TransmissionQuiescentUp,
    TransmissionQuiescentDown,
    TransmissionQuiescentLeft,
    TransmissionQuiescentRight,

    // Special transmission states
    SpecialTransmissionExcitedUp,
    SpecialTransmissionExcitedDown,
    SpecialTransmissionExcitedLeft,
    SpecialTransmissionExcitedRight,
    SpecialTransmissionQuiescentUp,
    SpecialTransmissionQuiescentDown,
    SpecialTransmissionQuiescentLeft,
    SpecialTransmissionQuiescentRight,
}

impl Display for VonNeumannState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Very imperfect, but GUI is the primary tool for working with this automaton
        let char = match &self {
            Self::Ground => "__",
            Self::Sensitized => "S_",
            Self::Sensitized0 => "S0",
            Self::Sensitized00 => "00",
            Self::Sensitized000 => "OO",
            Self::Sensitized01 => "01",
            Self::Sensitized1 => "S1",
            Self::Sensitized10 => "10",
            Self::Sensitized11 => "11",
            Self::Confluent00 => "C0",
            Self::Confluent01 => "C1",
            Self::Confluent10 => "C2",
            Self::Confluent11 => "C3",
            Self::TransmissionExcitedUp => "^^",
            Self::TransmissionExcitedDown => "vv",
            Self::TransmissionExcitedLeft => "<<",
            Self::TransmissionExcitedRight => ">>",
            Self::TransmissionQuiescentUp => "^_",
            Self::TransmissionQuiescentDown => "v_",
            Self::TransmissionQuiescentLeft => "<_",
            Self::TransmissionQuiescentRight => ">_",
            Self::SpecialTransmissionExcitedUp => "^!",
            Self::SpecialTransmissionExcitedDown => "v!",
            Self::SpecialTransmissionExcitedLeft => "<!",
            Self::SpecialTransmissionExcitedRight => ">!",
            Self::SpecialTransmissionQuiescentUp => "!^",
            Self::SpecialTransmissionQuiescentDown => "!v",
            Self::SpecialTransmissionQuiescentLeft => "!<",
            Self::SpecialTransmissionQuiescentRight => "!>",
        };
        f.write_str(char)
    }
}

impl CellStateVisuals for VonNeumannState {
    fn glyph_svg(self) -> Option<&'static str> {
        match self {
            Self::Ground => None,
            Self::Sensitized => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized0 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized00 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized000 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized01 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized1 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized10 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Sensitized11 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Confluent00 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Confluent01 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Confluent10 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::Confluent11 => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::TransmissionExcitedUp => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::TransmissionExcitedDown => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::TransmissionExcitedLeft => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::TransmissionExcitedRight => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::TransmissionQuiescentUp => Some(include_str!("../game_of_life/glyphs/live.svg")),
            Self::TransmissionQuiescentDown => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::TransmissionQuiescentLeft => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::TransmissionQuiescentRight => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionExcitedUp => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionExcitedDown => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionExcitedLeft => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionExcitedRight => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionQuiescentUp => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionQuiescentDown => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionQuiescentLeft => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
            Self::SpecialTransmissionQuiescentRight => {
                Some(include_str!("../game_of_life/glyphs/live.svg"))
            }
        }
    }

    fn pixel_color(self) -> Option<[u8; 3]> {
        match self {
            Self::Ground => None,
            Self::Sensitized => Some([32, 33, 36]),
            Self::Sensitized0 => Some([32, 33, 36]),
            Self::Sensitized00 => Some([32, 33, 36]),
            Self::Sensitized000 => Some([32, 33, 36]),
            Self::Sensitized01 => Some([32, 33, 36]),
            Self::Sensitized1 => Some([32, 33, 36]),
            Self::Sensitized10 => Some([32, 33, 36]),
            Self::Sensitized11 => Some([32, 33, 36]),
            Self::Confluent00 => Some([32, 33, 36]),
            Self::Confluent01 => Some([32, 33, 36]),
            Self::Confluent10 => Some([32, 33, 36]),
            Self::Confluent11 => Some([32, 33, 36]),
            Self::TransmissionExcitedUp => Some([32, 33, 36]),
            Self::TransmissionExcitedDown => Some([32, 33, 36]),
            Self::TransmissionExcitedLeft => Some([32, 33, 36]),
            Self::TransmissionExcitedRight => Some([32, 33, 36]),
            Self::TransmissionQuiescentUp => Some([32, 33, 36]),
            Self::TransmissionQuiescentDown => Some([32, 33, 36]),
            Self::TransmissionQuiescentLeft => Some([32, 33, 36]),
            Self::TransmissionQuiescentRight => Some([32, 33, 36]),
            Self::SpecialTransmissionExcitedUp => Some([32, 33, 36]),
            Self::SpecialTransmissionExcitedDown => Some([32, 33, 36]),
            Self::SpecialTransmissionExcitedLeft => Some([32, 33, 36]),
            Self::SpecialTransmissionExcitedRight => Some([32, 33, 36]),
            Self::SpecialTransmissionQuiescentUp => Some([32, 33, 36]),
            Self::SpecialTransmissionQuiescentDown => Some([32, 33, 36]),
            Self::SpecialTransmissionQuiescentLeft => Some([32, 33, 36]),
            Self::SpecialTransmissionQuiescentRight => Some([32, 33, 36]),
        }
    }
}

#[derive(Default)]
pub struct VonNeumannEvaluator;

#[derive(PartialEq, Eq, Clone, Copy)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    #[inline]
    pub fn invert(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum TransmissionKind {
    Normal,
    Special,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum TransmissionState {
    Quiescent,
    Excited,
}

impl VonNeumannState {
    #[inline]
    fn decode_transmission(self) -> Option<(TransmissionKind, TransmissionState, Direction)> {
        match self {
            VonNeumannState::TransmissionExcitedUp => Some((
                TransmissionKind::Normal,
                TransmissionState::Excited,
                Direction::Up,
            )),
            VonNeumannState::TransmissionExcitedDown => Some((
                TransmissionKind::Normal,
                TransmissionState::Excited,
                Direction::Down,
            )),
            VonNeumannState::TransmissionExcitedLeft => Some((
                TransmissionKind::Normal,
                TransmissionState::Excited,
                Direction::Left,
            )),
            VonNeumannState::TransmissionExcitedRight => Some((
                TransmissionKind::Normal,
                TransmissionState::Excited,
                Direction::Right,
            )),
            VonNeumannState::TransmissionQuiescentUp => Some((
                TransmissionKind::Normal,
                TransmissionState::Quiescent,
                Direction::Up,
            )),
            VonNeumannState::TransmissionQuiescentDown => Some((
                TransmissionKind::Normal,
                TransmissionState::Quiescent,
                Direction::Down,
            )),
            VonNeumannState::TransmissionQuiescentLeft => Some((
                TransmissionKind::Normal,
                TransmissionState::Quiescent,
                Direction::Left,
            )),
            VonNeumannState::TransmissionQuiescentRight => Some((
                TransmissionKind::Normal,
                TransmissionState::Quiescent,
                Direction::Right,
            )),
            VonNeumannState::SpecialTransmissionExcitedUp => Some((
                TransmissionKind::Special,
                TransmissionState::Excited,
                Direction::Up,
            )),
            VonNeumannState::SpecialTransmissionExcitedDown => Some((
                TransmissionKind::Special,
                TransmissionState::Excited,
                Direction::Down,
            )),
            VonNeumannState::SpecialTransmissionExcitedLeft => Some((
                TransmissionKind::Special,
                TransmissionState::Excited,
                Direction::Left,
            )),
            VonNeumannState::SpecialTransmissionExcitedRight => Some((
                TransmissionKind::Special,
                TransmissionState::Excited,
                Direction::Right,
            )),
            VonNeumannState::SpecialTransmissionQuiescentUp => Some((
                TransmissionKind::Special,
                TransmissionState::Quiescent,
                Direction::Up,
            )),
            VonNeumannState::SpecialTransmissionQuiescentDown => Some((
                TransmissionKind::Special,
                TransmissionState::Quiescent,
                Direction::Down,
            )),
            VonNeumannState::SpecialTransmissionQuiescentLeft => Some((
                TransmissionKind::Special,
                TransmissionState::Quiescent,
                Direction::Left,
            )),
            VonNeumannState::SpecialTransmissionQuiescentRight => Some((
                TransmissionKind::Special,
                TransmissionState::Quiescent,
                Direction::Right,
            )),
            _ => None,
        }
    }

    #[inline]
    fn encode_transmission(
        kind: TransmissionKind,
        state: TransmissionState,
        direction: Direction,
    ) -> VonNeumannState {
        match (kind, state, direction) {
            (TransmissionKind::Normal, TransmissionState::Excited, Direction::Up) => {
                VonNeumannState::TransmissionExcitedUp
            }
            (TransmissionKind::Normal, TransmissionState::Excited, Direction::Down) => {
                VonNeumannState::TransmissionExcitedDown
            }
            (TransmissionKind::Normal, TransmissionState::Excited, Direction::Left) => {
                VonNeumannState::TransmissionExcitedLeft
            }
            (TransmissionKind::Normal, TransmissionState::Excited, Direction::Right) => {
                VonNeumannState::TransmissionExcitedRight
            }
            (TransmissionKind::Normal, TransmissionState::Quiescent, Direction::Up) => {
                VonNeumannState::TransmissionQuiescentUp
            }
            (TransmissionKind::Normal, TransmissionState::Quiescent, Direction::Down) => {
                VonNeumannState::TransmissionQuiescentDown
            }
            (TransmissionKind::Normal, TransmissionState::Quiescent, Direction::Left) => {
                VonNeumannState::TransmissionQuiescentLeft
            }
            (TransmissionKind::Normal, TransmissionState::Quiescent, Direction::Right) => {
                VonNeumannState::TransmissionQuiescentRight
            }
            (TransmissionKind::Special, TransmissionState::Excited, Direction::Up) => {
                VonNeumannState::SpecialTransmissionExcitedUp
            }
            (TransmissionKind::Special, TransmissionState::Excited, Direction::Down) => {
                VonNeumannState::SpecialTransmissionExcitedDown
            }
            (TransmissionKind::Special, TransmissionState::Excited, Direction::Left) => {
                VonNeumannState::SpecialTransmissionExcitedLeft
            }
            (TransmissionKind::Special, TransmissionState::Excited, Direction::Right) => {
                VonNeumannState::SpecialTransmissionExcitedRight
            }
            (TransmissionKind::Special, TransmissionState::Quiescent, Direction::Up) => {
                VonNeumannState::SpecialTransmissionQuiescentUp
            }
            (TransmissionKind::Special, TransmissionState::Quiescent, Direction::Down) => {
                VonNeumannState::SpecialTransmissionQuiescentDown
            }
            (TransmissionKind::Special, TransmissionState::Quiescent, Direction::Left) => {
                VonNeumannState::SpecialTransmissionQuiescentLeft
            }
            (TransmissionKind::Special, TransmissionState::Quiescent, Direction::Right) => {
                VonNeumannState::SpecialTransmissionQuiescentRight
            }
        }
    }

    #[inline]
    fn decode_confluent(self) -> Option<(bool, bool)> {
        match self {
            Self::Confluent00 => Some((false, false)),
            Self::Confluent01 => Some((false, true)),
            Self::Confluent10 => Some((true, false)),
            Self::Confluent11 => Some((true, true)),
            _ => None,
        }
    }

    #[inline]
    fn encode_confluent(is_excited: bool, will_be_excited: bool) -> VonNeumannState {
        match (is_excited, will_be_excited) {
            (false, false) => VonNeumannState::Confluent00,
            (false, true) => VonNeumannState::Confluent01,
            (true, false) => VonNeumannState::Confluent10,
            (true, true) => VonNeumannState::Confluent11,
        }
    }
}

impl CellRuleEvaluator<VonNeumannState, VonNeumannNeighborhood<VonNeumannState>>
    for VonNeumannEvaluator
{
    fn evaluate(
        &self,
        state: VonNeumannState,
        neighbors: &VonNeumannNeighborhood<VonNeumannState>,
    ) -> VonNeumannState {
        // Transmission rules
        if let Some((kind, _, direction)) = state.decode_transmission() {
            // Collect all incoming transmissions
            // TODO: Maybe add neighbor iterating with direction?
            let mut incoming_transmissions = vec![];
            if let Some((nkind, nstate, ndirection)) = neighbors.up().decode_transmission()
                && ndirection == Direction::Down
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.down().decode_transmission()
                && ndirection == Direction::Up
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.left().decode_transmission()
                && ndirection == Direction::Right
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.right().decode_transmission()
                && ndirection == Direction::Left
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }

            // Destruction rule
            for &(nkind, nstate, _) in incoming_transmissions.iter() {
                if nkind != kind && nstate == TransmissionState::Excited {
                    return VonNeumannState::Ground;
                }
            }

            // Transmission rule
            for (nkind, nstate, ndirection) in incoming_transmissions {
                if nkind == kind
                    && nstate == TransmissionState::Excited
                    && direction.invert() != ndirection
                {
                    return VonNeumannState::encode_transmission(
                        kind,
                        TransmissionState::Excited,
                        direction,
                    );
                }
            }

            // Transmission from confluent cells
            let mut adjacent_confluents = vec![];
            if direction != Direction::Up
                && let Some((is_excited, _)) = neighbors.up().decode_confluent()
            {
                adjacent_confluents.push(is_excited);
            }
            if direction != Direction::Down
                && let Some((is_excited, _)) = neighbors.down().decode_confluent()
            {
                adjacent_confluents.push(is_excited);
            }
            if direction != Direction::Left
                && let Some((is_excited, _)) = neighbors.left().decode_confluent()
            {
                adjacent_confluents.push(is_excited);
            }
            if direction != Direction::Right
                && let Some((is_excited, _)) = neighbors.right().decode_confluent()
            {
                adjacent_confluents.push(is_excited);
            }

            if adjacent_confluents.iter().any(|b| *b) {
                return VonNeumannState::encode_transmission(
                    kind,
                    TransmissionState::Excited,
                    direction,
                );
            }

            return VonNeumannState::encode_transmission(
                kind,
                TransmissionState::Quiescent,
                direction,
            );
        }

        // Confluent state rules
        if let Some((_, will_be_excited)) = state.decode_confluent() {
            let mut incoming_transmissions = vec![];
            if let Some((nkind, nstate, ndirection)) = neighbors.up().decode_transmission()
                && ndirection == Direction::Down
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.down().decode_transmission()
                && ndirection == Direction::Up
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.left().decode_transmission()
                && ndirection == Direction::Right
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.right().decode_transmission()
                && ndirection == Direction::Left
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }

            // Destruction rule
            if incoming_transmissions.iter().any(|&(nkind, nstate, _)| {
                nkind == TransmissionKind::Special && nstate == TransmissionState::Excited
            }) {
                return VonNeumannState::Ground;
            }

            // AND transmission rule
            let mut ordinary_incoming_transmissions = incoming_transmissions;
            ordinary_incoming_transmissions
                .retain(|&(nkind, _, _)| nkind == TransmissionKind::Normal);
            if !ordinary_incoming_transmissions.is_empty()
                && ordinary_incoming_transmissions
                    .into_iter()
                    .all(|(_, nstate, _)| nstate == TransmissionState::Excited)
            {
                return VonNeumannState::encode_confluent(will_be_excited, true);
            }

            return VonNeumannState::encode_confluent(will_be_excited, false);
        }

        // Sensitized transition rules
        {
            let mut incoming_transmissions = vec![];
            if let Some((nkind, nstate, ndirection)) = neighbors.up().decode_transmission()
                && ndirection == Direction::Down
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.down().decode_transmission()
                && ndirection == Direction::Up
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.left().decode_transmission()
                && ndirection == Direction::Right
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            if let Some((nkind, nstate, ndirection)) = neighbors.right().decode_transmission()
                && ndirection == Direction::Left
            {
                incoming_transmissions.push((nkind, nstate, ndirection));
            }
            let is_excited = incoming_transmissions
                .into_iter()
                .any(|(_, nstate, _)| nstate == TransmissionState::Excited);

            match state {
                VonNeumannState::Ground => {
                    if is_excited {
                        return VonNeumannState::Sensitized;
                    }
                }
                VonNeumannState::Sensitized => {
                    return if is_excited {
                        VonNeumannState::Sensitized1
                    } else {
                        VonNeumannState::Sensitized0
                    };
                }
                VonNeumannState::Sensitized0 => {
                    return if is_excited {
                        VonNeumannState::Sensitized01
                    } else {
                        VonNeumannState::Sensitized00
                    };
                }
                VonNeumannState::Sensitized00 => {
                    return if is_excited {
                        VonNeumannState::TransmissionQuiescentLeft
                    } else {
                        VonNeumannState::Sensitized000
                    };
                }
                VonNeumannState::Sensitized000 => {
                    return if is_excited {
                        VonNeumannState::TransmissionQuiescentUp
                    } else {
                        VonNeumannState::TransmissionQuiescentRight
                    };
                }
                VonNeumannState::Sensitized01 => {
                    return if is_excited {
                        VonNeumannState::SpecialTransmissionQuiescentRight
                    } else {
                        VonNeumannState::TransmissionQuiescentDown
                    };
                }
                VonNeumannState::Sensitized1 => {
                    return if is_excited {
                        VonNeumannState::Sensitized11
                    } else {
                        VonNeumannState::Sensitized10
                    };
                }
                VonNeumannState::Sensitized10 => {
                    return if is_excited {
                        VonNeumannState::SpecialTransmissionQuiescentLeft
                    } else {
                        VonNeumannState::SpecialTransmissionQuiescentUp
                    };
                }
                VonNeumannState::Sensitized11 => {
                    return if is_excited {
                        VonNeumannState::Confluent00
                    } else {
                        VonNeumannState::SpecialTransmissionQuiescentDown
                    };
                }
                _ => {}
            }
        }

        VonNeumannState::Ground
    }
}

pub struct VonNeumannConfig;

impl CellularAutomataConfig for VonNeumannConfig {
    const NAME: &'static str = "Von Neumann";
    type State = VonNeumannState;
    type Evaluator = VonNeumannEvaluator;
    type Neighborhood = VonNeumannNeighborhood<VonNeumannState>;
}

pub type VonNeumann = CellularAutomaton<VonNeumannConfig>;
