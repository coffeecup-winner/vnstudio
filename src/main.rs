mod automata;
mod core;
mod gui;

use gui::app::VnStudioApp;

fn main() -> eframe::Result<()> {
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
