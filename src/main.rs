mod automata;
mod core;
mod gui;

use gui::app::VnStudioApp;

use crate::automata::von_neumann::VonNeumann;

fn main() -> eframe::Result<()> {
    {
        let mut x = VonNeumann::new();
        let start = std::time::Instant::now();
        x.switch_to_lut();
        let end = std::time::Instant::now();
        println!("LUT building took {}ms", (end - start).as_millis());
    }

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "VNStudio",
        options,
        Box::new(|creation_context| Ok(Box::new(VnStudioApp::new(creation_context)))),
    )
}
