use eframe::egui::{Color32, Painter, Pos2, Rect, Shape, Stroke};
use roxmltree::Node;

#[derive(Clone, Debug)]
pub struct SvgGlyph {
    primitives: Vec<GlyphPrimitive>,
}

#[derive(Clone, Debug)]
enum GlyphPrimitive {
    Rect {
        min: [f32; 2],
        max: [f32; 2],
        fill: Option<Color32>,
        stroke: Option<Stroke>,
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        fill: Option<Color32>,
        stroke: Option<Stroke>,
    },
    Line {
        from: [f32; 2],
        to: [f32; 2],
        stroke: Stroke,
    },
    Polygon {
        points: Vec<[f32; 2]>,
        fill: Option<Color32>,
        stroke: Option<Stroke>,
    },
    Path {
        points: Vec<[f32; 2]>,
        closed: bool,
        fill: Option<Color32>,
        stroke: Option<Stroke>,
    },
}

#[derive(Clone, Copy, Debug)]
struct ViewBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

impl SvgGlyph {
    pub fn parse(svg: &str) -> Result<Self, String> {
        let document = roxmltree::Document::parse(svg).map_err(|err| err.to_string())?;
        let root = document
            .descendants()
            .find(|node| node.is_element() && node.tag_name().name() == "svg")
            .ok_or_else(|| "missing <svg> root".to_string())?;
        let view_box = parse_view_box(root.attribute("viewBox").unwrap_or("0 0 1 1"))?;
        let mut primitives = Vec::new();

        for node in root.descendants().filter(|node| node.is_element()) {
            if node == root {
                continue;
            }

            match node.tag_name().name() {
                "rect" => {
                    let x = parse_f32_attr(node, "x").unwrap_or(0.0);
                    let y = parse_f32_attr(node, "y").unwrap_or(0.0);
                    let width = parse_f32_attr(node, "width").unwrap_or(0.0);
                    let height = parse_f32_attr(node, "height").unwrap_or(0.0);
                    primitives.push(GlyphPrimitive::Rect {
                        min: view_box.normalize(x, y),
                        max: view_box.normalize(x + width, y + height),
                        fill: parse_fill(node),
                        stroke: parse_stroke(node),
                    });
                }
                "circle" => {
                    let cx = parse_f32_attr(node, "cx").unwrap_or(0.0);
                    let cy = parse_f32_attr(node, "cy").unwrap_or(0.0);
                    let radius = parse_f32_attr(node, "r").unwrap_or(0.0) / view_box.width;
                    primitives.push(GlyphPrimitive::Circle {
                        center: view_box.normalize(cx, cy),
                        radius,
                        fill: parse_fill(node),
                        stroke: parse_stroke(node),
                    });
                }
                "line" => {
                    if let Some(stroke) = parse_stroke(node) {
                        primitives.push(GlyphPrimitive::Line {
                            from: view_box.normalize(
                                parse_f32_attr(node, "x1").unwrap_or(0.0),
                                parse_f32_attr(node, "y1").unwrap_or(0.0),
                            ),
                            to: view_box.normalize(
                                parse_f32_attr(node, "x2").unwrap_or(0.0),
                                parse_f32_attr(node, "y2").unwrap_or(0.0),
                            ),
                            stroke,
                        });
                    }
                }
                "polygon" => {
                    if let Some(points) = node.attribute("points") {
                        primitives.push(GlyphPrimitive::Polygon {
                            points: parse_points(points, view_box)?,
                            fill: parse_fill(node),
                            stroke: parse_stroke(node),
                        });
                    }
                }
                "path" => {
                    if let Some(data) = node.attribute("d") {
                        let (points, closed) = parse_simple_path(data, view_box)?;
                        primitives.push(GlyphPrimitive::Path {
                            points,
                            closed,
                            fill: parse_fill(node),
                            stroke: parse_stroke(node),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(Self { primitives })
    }

    pub fn paint(&self, painter: &Painter, rect: Rect) {
        for primitive in &self.primitives {
            primitive.paint(painter, rect);
        }
    }
}

impl GlyphPrimitive {
    fn paint(&self, painter: &Painter, rect: Rect) {
        match self {
            GlyphPrimitive::Rect {
                min,
                max,
                fill,
                stroke,
            } => {
                let rect = Rect::from_min_max(map_point(rect, *min), map_point(rect, *max));
                if let Some(fill) = fill {
                    painter.rect_filled(rect, 0.0, *fill);
                }
                if let Some(stroke) = stroke {
                    painter.line_segment([rect.left_top(), rect.right_top()], *stroke);
                    painter.line_segment([rect.right_top(), rect.right_bottom()], *stroke);
                    painter.line_segment([rect.right_bottom(), rect.left_bottom()], *stroke);
                    painter.line_segment([rect.left_bottom(), rect.left_top()], *stroke);
                }
            }
            GlyphPrimitive::Circle {
                center,
                radius,
                fill,
                stroke,
            } => {
                let center = map_point(rect, *center);
                let radius = radius * rect.width().min(rect.height());
                if let Some(fill) = fill {
                    painter.circle_filled(center, radius, *fill);
                }
                if let Some(stroke) = stroke {
                    painter.circle_stroke(center, radius, *stroke);
                }
            }
            GlyphPrimitive::Line { from, to, stroke } => {
                painter.line_segment([map_point(rect, *from), map_point(rect, *to)], *stroke);
            }
            GlyphPrimitive::Polygon {
                points,
                fill,
                stroke,
            } => paint_points(painter, rect, points, true, *fill, *stroke),
            GlyphPrimitive::Path {
                points,
                closed,
                fill,
                stroke,
            } => paint_points(painter, rect, points, *closed, *fill, *stroke),
        }
    }
}

impl ViewBox {
    fn normalize(self, x: f32, y: f32) -> [f32; 2] {
        [(x - self.x) / self.width, (y - self.y) / self.height]
    }
}

fn paint_points(
    painter: &Painter,
    rect: Rect,
    points: &[[f32; 2]],
    closed: bool,
    fill: Option<Color32>,
    stroke: Option<Stroke>,
) {
    if points.len() < 2 {
        return;
    }

    let mapped: Vec<Pos2> = points.iter().map(|point| map_point(rect, *point)).collect();
    if let Some(fill) = fill
        && closed
        && points.len() >= 3
    {
        painter.add(Shape::convex_polygon(
            mapped.clone(),
            fill,
            stroke.unwrap_or(Stroke::NONE),
        ));
        return;
    }

    if let Some(stroke) = stroke {
        for segment in mapped.windows(2) {
            painter.line_segment([segment[0], segment[1]], stroke);
        }
        if closed {
            painter.line_segment([mapped[mapped.len() - 1], mapped[0]], stroke);
        }
    }
}

fn map_point(rect: Rect, point: [f32; 2]) -> Pos2 {
    Pos2::new(
        rect.left() + point[0] * rect.width(),
        rect.top() + point[1] * rect.height(),
    )
}

fn parse_view_box(value: &str) -> Result<ViewBox, String> {
    let values: Vec<f32> = value
        .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
        .filter(|part| !part.is_empty())
        .map(parse_svg_number)
        .collect::<Result<_, _>>()?;

    if values.len() != 4 || values[2] <= 0.0 || values[3] <= 0.0 {
        return Err("viewBox must contain four positive numeric values".to_string());
    }

    Ok(ViewBox {
        x: values[0],
        y: values[1],
        width: values[2],
        height: values[3],
    })
}

fn parse_points(value: &str, view_box: ViewBox) -> Result<Vec<[f32; 2]>, String> {
    let values: Vec<f32> = value
        .split(|ch: char| ch.is_ascii_whitespace() || ch == ',')
        .filter(|part| !part.is_empty())
        .map(parse_svg_number)
        .collect::<Result<_, _>>()?;

    if !values.len().is_multiple_of(2) {
        return Err("polygon points must contain x/y pairs".to_string());
    }

    Ok(values
        .chunks_exact(2)
        .map(|chunk| view_box.normalize(chunk[0], chunk[1]))
        .collect())
}

fn parse_simple_path(value: &str, view_box: ViewBox) -> Result<(Vec<[f32; 2]>, bool), String> {
    let tokens = tokenize_path(value);
    let mut points = Vec::new();
    let mut current_command = 'M';
    let mut cursor = [0.0, 0.0];
    let mut i = 0;
    let mut closed = false;

    while i < tokens.len() {
        if let PathToken::Command(command) = tokens[i] {
            current_command = command;
            i += 1;
        }

        match current_command {
            'M' | 'L' => {
                let x = read_number(&tokens, &mut i)?;
                let y = read_number(&tokens, &mut i)?;
                cursor = [x, y];
                points.push(view_box.normalize(x, y));
                if current_command == 'M' {
                    current_command = 'L';
                }
            }
            'm' | 'l' => {
                let x = cursor[0] + read_number(&tokens, &mut i)?;
                let y = cursor[1] + read_number(&tokens, &mut i)?;
                cursor = [x, y];
                points.push(view_box.normalize(x, y));
                if current_command == 'm' {
                    current_command = 'l';
                }
            }
            'H' => {
                cursor[0] = read_number(&tokens, &mut i)?;
                points.push(view_box.normalize(cursor[0], cursor[1]));
            }
            'h' => {
                cursor[0] += read_number(&tokens, &mut i)?;
                points.push(view_box.normalize(cursor[0], cursor[1]));
            }
            'V' => {
                cursor[1] = read_number(&tokens, &mut i)?;
                points.push(view_box.normalize(cursor[0], cursor[1]));
            }
            'v' => {
                cursor[1] += read_number(&tokens, &mut i)?;
                points.push(view_box.normalize(cursor[0], cursor[1]));
            }
            'Z' | 'z' => {
                closed = true;
                i += usize::from(i < tokens.len() && matches!(tokens[i], PathToken::Command(_)));
            }
            command => return Err(format!("unsupported path command: {command}")),
        }
    }

    Ok((points, closed))
}

#[derive(Clone, Copy, Debug)]
enum PathToken {
    Command(char),
    Number(f32),
}

fn tokenize_path(value: &str) -> Vec<PathToken> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = value.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];
        if ch.is_ascii_whitespace() || ch == ',' {
            i += 1;
        } else if ch.is_ascii_alphabetic() {
            tokens.push(PathToken::Command(ch));
            i += 1;
        } else {
            let start = i;
            i += 1;
            while i < chars.len() {
                let ch = chars[i];
                if ch.is_ascii_digit()
                    || matches!(ch, '.' | 'e' | 'E')
                    || (matches!(ch, '-' | '+') && matches!(chars[i - 1], 'e' | 'E'))
                {
                    i += 1;
                } else {
                    break;
                }
            }
            let text: String = chars[start..i].iter().collect();
            if let Ok(number) = parse_svg_number(&text) {
                tokens.push(PathToken::Number(number));
            }
        }
    }

    tokens
}

fn read_number(tokens: &[PathToken], index: &mut usize) -> Result<f32, String> {
    match tokens.get(*index) {
        Some(PathToken::Number(number)) => {
            *index += 1;
            Ok(*number)
        }
        _ => Err("expected path number".to_string()),
    }
}

fn parse_f32_attr(node: Node<'_, '_>, attribute: &str) -> Option<f32> {
    node.attribute(attribute)
        .and_then(|value| parse_svg_number(value).ok())
}

fn parse_svg_number(value: &str) -> Result<f32, String> {
    let trimmed = value.trim().trim_end_matches("px");
    trimmed
        .parse::<f32>()
        .map_err(|_| format!("invalid numeric value: {value}"))
}

fn parse_fill(node: Node<'_, '_>) -> Option<Color32> {
    parse_color(node.attribute("fill").unwrap_or("#000000"))
}

fn parse_stroke(node: Node<'_, '_>) -> Option<Stroke> {
    let color = parse_color(node.attribute("stroke")?)?;
    let width = parse_f32_attr(node, "stroke-width").unwrap_or(0.02);
    Some(Stroke::new(width, color))
}

fn parse_color(value: &str) -> Option<Color32> {
    let value = value.trim();
    if value == "none" {
        return None;
    }

    let hex = value.strip_prefix('#')?;
    let parse_pair = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16).ok();
    match hex.len() {
        6 => Some(Color32::from_rgb(
            parse_pair(0..2)?,
            parse_pair(2..4)?,
            parse_pair(4..6)?,
        )),
        8 => Some(Color32::from_rgba_premultiplied(
            parse_pair(0..2)?,
            parse_pair(2..4)?,
            parse_pair(4..6)?,
            parse_pair(6..8)?,
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_circle() {
        let glyph = SvgGlyph::parse(
            r##"<svg viewBox="0 0 1 1"><circle cx="0.5" cy="0.5" r="0.25" fill="#000000"/></svg>"##,
        )
        .unwrap();

        assert_eq!(glyph.primitives.len(), 1);
    }
}
