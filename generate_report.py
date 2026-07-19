"""
Generate a PDF report explaining the DQD simulation pipeline.
Run: python generate_report.py
Output: docs/program_report.pdf
"""
import os
from fpdf import FPDF

OUTPUT = os.path.join(os.path.dirname(__file__), "docs", "program_report.pdf")


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "DQD Simulation Pipeline - Program Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 180)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # ?? helpers ?????????????????????????????????????????????????????????????
    def chapter_title(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(30, 80, 160)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def section_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 80, 160)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 80, 160)
        self.line(self.l_margin, self.get_y(), self.l_margin + 80, self.get_y())
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body(self, text):
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, items):
        self.set_font("Helvetica", size=10)
        w = self.w - self.l_margin - self.r_margin
        for item in items:
            self.multi_cell(w, 6, f"   -  {item}")
        self.ln(1)

    def code_block(self, lines):
        self.set_font("Courier", size=9)
        self.set_fill_color(240, 240, 240)
        self.set_draw_color(200, 200, 200)
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        for line in lines:
            self.cell(w, 5.5, f"  {line}", fill=True, border=0, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", size=10)
        self.ln(3)

    def two_col_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [60, 120]
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(220, 230, 245)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, f" {h}", border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", size=9)
        fill = False
        self.set_fill_color(248, 248, 248)
        for row in rows:
            for val, w in zip(row, col_widths):
                self.multi_cell(w, 6, f" {val}", border=1, fill=fill, new_x="RIGHT", new_y="LAST")
            self.ln()
            fill = not fill
        self.ln(3)


# ?? build PDF ????????????????????????????????????????????????????????????????
pdf = PDF()
pdf.set_margins(20, 20, 20)
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ?? TITLE PAGE ???????????????????????????????????????????????????????????????
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 80, 160)
pdf.ln(20)
pdf.cell(0, 12, "DQD Simulation Pipeline", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 10, "Program Structure & How It Works", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", size=11)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "Project: static_RBC_test_noise_3_17", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(25)

pdf.set_font("Helvetica", size=10)
pdf.set_text_color(0, 0, 0)
pdf.set_draw_color(200, 200, 200)
pdf.set_fill_color(245, 248, 255)
pdf.rect(pdf.l_margin, pdf.get_y(), pdf.w - pdf.l_margin - pdf.r_margin, 50, "DF")
pdf.set_xy(pdf.l_margin + 5, pdf.get_y() + 5)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 7, "What this program does:", new_x="LMARGIN", new_y="NEXT")
pdf.set_x(pdf.l_margin + 5)
pdf.set_font("Helvetica", size=10)
pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 10, 6,
    "This software generates synthetic Double Quantum Dot (DQD) charge-sensing training data. "
    "It simulates quantum dot physics, performs ray-based peak detection on stability diagrams, "
    "crops and analyzes detected peaks, binarizes charge-sensor outputs, and produces fully "
    "annotated datasets for machine learning.")
pdf.ln(2)

# ?? PAGE 2: OVERVIEW ?????????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title("1. Project Overview")

pdf.section_title("What is a Double Quantum Dot (DQD)?")
pdf.body(
    "A Double Quantum Dot is a nanoscale device used in quantum computing. It consists of two "
    "coupled quantum dots, each capable of trapping individual electrons. The stability diagram "
    "(charge-stability diagram) maps out regions of stable charge configurations as a function "
    "of two gate voltages (Vx, Vy). The boundaries between these regions appear as sharp lines "
    "called charge-transition lines. This program simulates those diagrams and trains models to "
    "detect the transition lines automatically."
)

pdf.section_title("Goal of the Pipeline")
pdf.bullet([
    "Generate many randomized DQD simulations (each with different capacitance parameters).",
    "Compute charge-sensing output + add realistic noise.",
    "Detect peaks (charge transitions) along rays swept at different angles.",
    "Crop regions around detected peaks and run detailed sweep analysis.",
    "Produce binary ground-truth labels and annotated images for ML training.",
])

pdf.section_title("How to Run")
pdf.body("Open scripts/run_simulation.py, set your parameters in main(), then run the script:")
pdf.code_block([
    "pipeline = DatasetPipeline(",
    "    base_save_dir='training_data',",
    "    n_samples=30,         # number of simulations",
    "    num_angles=2,         # ray angles between 0 and 90 degrees",
    "    ray_resolution=50,    # points sampled per ray",
    "    x_resolution=100,     # voltage grid size (100x100)",
    "    y_resolution=100,",
    "    crop_size=2.0,        # mV half-width around each peak",
    "    noise_std=0.01,       # white noise added to sensor",
    ")",
    "pipeline.run()",
])

pdf.section_title("Key Parameters")
pdf.two_col_table(
    ["Parameter", "Meaning"],
    [
        ["n_samples", "How many independent DQD simulations to generate"],
        ["num_angles", "Number of ray directions swept (spaced between 0° and 90°)"],
        ["ray_resolution", "How many voltage points are sampled along each ray"],
        ["x_resolution / y_resolution", "Grid size of the full stability diagram (e.g. 100x100)"],
        ["crop_size", "Half-width (mV) of the square cropped around each detected peak"],
        ["noise_std", "Standard deviation of Gaussian white noise added to charge sensor"],
        ["voltage_sweep", "Voltage range for Vx and Vy axes (default +/-1 V)"],
    ],
    col_widths=[55, 125],
)

# ?? PAGE 3: FOLDER STRUCTURE ?????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title("2. Source Code Structure")

pdf.body("The source code lives in src/dqd/ and is split into 7 sub-packages:")
pdf.code_block([
    "src/dqd/",
    "  config/              <- capacitance parameter intervals",
    "  simulation/          <- physics engine (DQD + capacitance matrices)",
    "  analysis/            <- ray detection, peak sweeps, binarization",
    "  coordinates/         <- voltage <-> pixel <-> local conversions",
    "  data/                <- file management, cropping, summary writing",
    "  visualization/       <- all plots, overlays, 3-D surfaces",
    "  pipeline/            <- master orchestrator (DatasetPipeline)",
    "",
    "scripts/",
    "  run_simulation.py    <- entry point (just calls DatasetPipeline)",
])

pdf.section_title("Module Responsibilities")
pdf.two_col_table(
    ["Module / File", "Responsibility"],
    [
        ["config/capacitance_config.py", "Defines min/max intervals for every capacitance matrix element (Cdd, Cgd, Cds, Cgs)"],
        ["simulation/matrix_generator.py", "Randomly draws capacitance values from config intervals (reproducible with seed)"],
        ["simulation/dqd_simulator.py", "Runs the full DQD physics simulation via qarray; saves .npy arrays and .jpg images"],
        ["analysis/ray_processor.py", "Fires rays at configured angles; detects peaks along each ray using scipy.find_peaks"],
        ["analysis/peak_detector.py", "Row-major directional sweeps (bottom-up, top-down) on cropped charge-sensor data"],
        ["analysis/binarizer.py", "Converts charge-sensor array to binary local-maxima image (ground-truth labels)"],
        ["coordinates/coordinate_mapper.py", "Converts between voltage (mV), pixel (0-based), and local (1-based) indices"],
        ["data/sample.py", "Moves raw simulation outputs into the canonical directory layout after each run"],
        ["data/cropped_region.py", "Extracts a square crop around a voltage-space peak; saves .npy + grid mapping"],
        ["data/summary_writer.py", "Aggregates sweep result files into summary_local.txt with scanned cells + peaks"],
        ["visualization/plotter.py", "Generates 3-D surfaces, ray scatter plots, current-vs-distance profiles"],
        ["visualization/overlay.py", "Renders scanned cells and peak markers onto charge-sensor background images"],
        ["pipeline/dataset_pipeline.py", "Orchestrates all steps above for N samples; the top-level controller"],
    ],
    col_widths=[65, 115],
)

# ?? PAGE 4: DATA FLOW ????????????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title("3. How One Sample is Generated (Data Flow)")

pdf.body(
    "Each call to pipeline.run() loops over N samples. For each sample, the following "
    "steps happen in order:"
)

steps = [
    ("Step 1 - Capacitance Matrices",
     "CapacitanceMatrixGenerator draws random values from CapacitanceConfig intervals "
     "to produce four matrices: Cdd (dot-to-dot), Cgd (gate-to-dot), Cds (dot-to-source), "
     "Cgs (gate-to-source). These define the physics of this particular simulated device."),

    ("Step 2 - DQD Simulation",
     "DQDSimulator passes the matrices to qarray (an external quantum dot library). "
     "It sweeps the gate voltages Vx and Vy over a 100x100 grid (or your chosen resolution) "
     "in the range +/-1 V. It computes:\n"
     "  - charge_sensing_data.npy  (raw charge sensor current)\n"
     "  - charge_sensing_noisy_z_data.npy  (same + white Gaussian noise)\n"
     "  - charge_sensing_grad_data.npy  (gradient of the above)\n"
     "  - double_dot_data.npy  (direct double-dot stability diagram)\n"
     "It also saves .jpg images and a hyperparameters.json file."),

    ("Step 3 - Ray Processing",
     "RayProcessor loads the noisy charge-sensing .npy file. It fires num_angles rays "
     "starting from the corner (max_x, max_y), each pointing inward at a different angle "
     "between 0° and 90°. Along each ray it samples ray_resolution points and calls "
     "scipy.signal.find_peaks to locate charge-transition peaks. Results are saved as "
     "peaks.json (flat list) and peaks_paired.json (grouped by angle)."),

    ("Step 4 - Directory Organization",
     "Sample.organize() moves the raw output files into a clean layout:\n"
     "  numpy/simulation/  <- all .npy data arrays\n"
     "  images/            <- all .jpg / .png visualizations"),

    ("Step 5 - Per-Peak Processing",
     "For every detected peak (Vx, Vy), the pipeline:\n"
     "  a) CroppedRegion: crops a 2 mV window around the peak from both charge_sensing and "
     "double_dot arrays; saves cropped .npy + grid mapping + center_indices.json\n"
     "  b) OverlayRenderer: generates a binary ground-truth label array (local maxima)\n"
     "  c) Plotter: draws a 3-D normalized surface plot of the cropped region\n"
     "  d) PeakDetector: runs two directional sweeps (bottom-up-right-left and "
     "top-down-left-right) and saves sweep_params_*.txt + animated .gif files\n"
     "  e) SummaryWriter: merges both sweep files into summary_local.txt\n"
     "  f) ChargeSensorBinarizer: creates binary_no_overlay.png and binary_with_overlay.png\n"
     "  g) CoordinateMapper: converts local 1-based row/col to voltage_coordinates.txt\n"
     "  h) OverlayRenderer: highlights this peak's scanned region on the full image"),

    ("Step 6 - Sample-Level Overlays",
     "After all peaks are processed, sample-level summary images are created:\n"
     "  - all_rays_peaks_overlay.png  (all ray paths + peak markers, colored by angle)\n"
     "  - all_scanned_peaks_overlay.png  (all scanned cells + peaks on full image)\n"
     "  - all_scanned_peaks_overlay.txt  (text-only version)\n"
     "  - summary.png  (2-D grid visualization of binary results)"),
]

for title, text in steps:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(230, 238, 255)
    pdf.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.set_x(pdf.l_margin + 4)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, text)
    pdf.ln(2)

# ?? PAGE 5: OUTPUT STRUCTURE ?????????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title("4. Output Directory Structure")

pdf.body("After running, each sample is saved in training_data/sample_N/:")
pdf.code_block([
    "training_data/",
    "  sample_1/",
    "  sample_2/",
    "  ...",
    "  sample_N/",
    "    hyperparameters.json           <- all simulation parameters used",
    "    peaks.json                     <- detected peaks [(vx, vy), ...]",
    "    peaks_paired.json              <- peaks grouped by ray angle",
    "    all_rays_peaks_overlay.png     <- all rays drawn on charge-sensor image",
    "    all_scanned_peaks_overlay.png  <- all scanned cells + peaks overlay",
    "    all_scanned_peaks_overlay.txt  <- text-only version of above",
    "    summary.png                    <- 2-D binary grid visualization",
    "    numpy/",
    "      simulation/",
    "        charge_sensing_data.npy",
    "        charge_sensing_noisy_z_data.npy",
    "        charge_sensing_grad_data.npy",
    "        double_dot_data.npy",
    "        ground_truth_labels.npy    <- binary local-maxima label array",
    "    images/",
    "      charge_sensing.jpg",
    "      double_dot_stability_diagram.jpg",
    "      original.png, rays_only.png, combined_plot.png, ...",
    "    cropped_results/",
    "      ray_0/                       <- results for first ray angle",
    "        peak_1/                    <- first detected peak on that ray",
    "          charge_sensing_cropped.npy",
    "          double_dot_cropped.npy",
    "          charge_sensing_cropped_grid_mapping.txt",
    "          center_indices.json",
    "          summary_local.txt",
    "          voltage_coordinates.txt",
    "          binary_no_overlay.png",
    "          binary_with_overlay.png",
    "          normalized_3d_surface.png",
    "          sweep_params_bottom_up_right_left.txt",
    "          sweep_params_top_down_left_right.txt",
    "          sweep_animation_*.gif",
    "      ray_1/ ...",
])

# ?? PAGE 6: COORDINATE SYSTEMS ???????????????????????????????????????????????
pdf.add_page()
pdf.chapter_title("5. Coordinate Systems")

pdf.body(
    "Three coordinate systems are used throughout the pipeline. Understanding them is "
    "key to reading the output files correctly."
)

pdf.two_col_table(
    ["Coordinate System", "Description"],
    [
        ["Voltage space  (Vx, Vy)",
         "Physical gate voltages in millivolts. Range is typically -1 V to +1 V on both axes. "
         "Used in peaks.json, voltage_coordinates.txt, and all physics computations."],
        ["Pixel / Global (row, col)",
         "0-based integer indices into the full simulation grid. Row 0 = top, Col 0 = left. "
         "Grid size = x_resolution x y_resolution (e.g. 100x100). Used internally and in "
         "center_indices.json."],
        ["Local / Cropped (row, col)",
         "1-based integer indices within a cropped region around a single peak. Row 1 = top "
         "of cropped window. Used in summary_local.txt, sweep_params_*.txt, and binarized images."],
    ],
    col_widths=[48, 132],
)

pdf.section_title("Conversion Functions (coordinate_mapper.py)")
pdf.two_col_table(
    ["Function", "Converts"],
    [
        ["voltage_to_pixel(vx, vy, ...)", "Voltage -> 0-based pixel index"],
        ["write_cropped_grid_mapping(...)", "Writes file mapping 1-based local row/col -> Vx/Vy"],
        ["local_to_voltage(summary, mapping, out)", "Reads local coords from summary, writes voltage_coordinates.txt"],
        ["extract_center_indices(json_path)", "Reads center pixel index from center_indices.json"],
    ],
    col_widths=[90, 90],
)

# ?? PAGE 7: KEY CLASSES QUICK REFERENCE ?????????????????????????????????????
pdf.add_page()
pdf.chapter_title("6. Key Classes - Quick Reference")

classes = [
    ("DatasetPipeline", "pipeline/dataset_pipeline.py",
     "Master controller. Loops over N samples. Calls all other classes in order. "
     "Constructor takes: base_save_dir, n_samples, num_angles, ray_resolution, "
     "x_resolution, y_resolution, crop_size, noise_std, voltage_sweep, config."),

    ("DQDSimulator", "simulation/dqd_simulator.py",
     "Physics engine. Receives params dict with capacitance matrices, voltage sweep, "
     "model params, and noise settings. Calls qarray internally. Method: run()."),

    ("CapacitanceMatrixGenerator", "simulation/matrix_generator.py",
     "Randomly samples capacitance values. Methods: generate_symmetric() for Cdd, "
     "generate_general() for Cgd/Cds/Cgs, generate_all(config) for all four at once."),

    ("RayProcessor", "analysis/ray_processor.py",
     "Ray-based peak detection. Constructor: (file_path_numpy, angles_deg, resolution, "
     "save_dir). Fires rays from (max_x, max_y) inward. Method: run() returns peaks dict."),

    ("PeakDetector", "analysis/peak_detector.py",
     "Directional sweep on cropped data. Scans rows from top and bottom, stopping at "
     "first local maximum per row. Method: run(data_path, hyperparams) returns sweep results."),

    ("CroppedRegion", "data/cropped_region.py",
     "Crops a voltage-space window around a peak. Methods: crop_and_save(file, x, y, size) "
     "and process_batch(base_dir, x, y, size) for both charge_sensing + double_dot files."),

    ("ChargeSensorBinarizer", "analysis/binarizer.py",
     "Converts charge-sensor 2-D array to binary image. Detects local maxima in both "
     "horizontal and vertical directions. Method: convert(data_path, ...)."),

    ("CoordinateMapper", "coordinates/coordinate_mapper.py",
     "All static methods. Handles all three coordinate systems. Key methods: "
     "voltage_to_pixel(), local_to_voltage(), write_cropped_grid_mapping()."),

    ("OverlayRenderer", "visualization/overlay.py",
     "All static methods. Draws scanned cells and peak markers on top of .npy-based images. "
     "Methods: highlight_voltage_coordinates(), highlight_all_rays_in_sample(), "
     "generate_ground_truth_array(), highlight_all_scanned_and_peaks_in_sample()."),

    ("Plotter", "visualization/plotter.py",
     "All static methods. Generates: 3-D normalized surface (plot_normalized_3d), ray "
     "scatter plots (plot_rays_with_peaks), current-vs-distance profiles, per-ray plots."),

    ("Sample", "data/sample.py",
     "File organizer for one sample directory. Method: organize() moves raw outputs into "
     "numpy/simulation/ and images/ subdirectories after simulation completes."),

    ("SummaryWriter", "data/summary_writer.py",
     "Reads sweep_params_*.txt files and writes a unified summary_local.txt with scanned "
     "cell locations, detected peak locations, and statistics. Method: write(directory, ...)."),

    ("CapacitanceConfig", "config/capacitance_config.py",
     "Dataclass holding interval dicts for Cdd, Cgd, Cds, Cgs. Methods: validate() and "
     "print_summary(). Pass a custom instance to DatasetPipeline to override defaults."),
]

for name, path, desc in classes:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 7, name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 5, f"  {path}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(pdf.l_margin + 4)
    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, desc)
    pdf.ln(2)

# ?? PAGE 8: EXTERNAL DEPENDENCIES ???????????????????????????????????????????
pdf.add_page()
pdf.chapter_title("7. External Dependencies")

pdf.two_col_table(
    ["Library", "Used For"],
    [
        ["qarray", "Core quantum dot physics simulation (ChargeSensedDotArray, DotArray). "
                   "This is the most critical external dependency - without it the simulation cannot run."],
        ["numpy", "All numerical array operations, saving/loading .npy files."],
        ["matplotlib", "All plots: stability diagrams, ray visualizations, 3-D surfaces, overlays."],
        ["scipy", "scipy.signal.find_peaks - detects peaks along rays and in sweep rows."],
        ["Pillow (PIL)", "Image loading and manipulation for overlay rendering."],
    ],
    col_widths=[35, 145],
)

pdf.section_title("Important Note on qarray")
pdf.body(
    "The qarray library must be installed for the simulation to run. It is an open-source "
    "Python package for simulating quantum dot charge stability diagrams. All capacitance "
    "matrices generated by CapacitanceMatrixGenerator are passed directly into qarray's "
    "ChargeSensedDotArray and DotArray classes."
)

pdf.chapter_title("8. Summary Diagram")
pdf.set_font("Courier", size=9)
pdf.set_fill_color(240, 240, 240)
diagram = [
    " run_simulation.py",
    "         |",
    "         v",
    " DatasetPipeline.run()   <-- n_samples=30, num_angles=2, ray_res=50, img_res=100",
    "         |",
    "         |--- [for each sample i = 1..N]",
    "         |",
    "         +--> CapacitanceMatrixGenerator  --> random Cdd, Cgd, Cds, Cgs",
    "         |",
    "         +--> DQDSimulator.run()          --> charge_sensing_data.npy",
    "         |                                    double_dot_data.npy",
    "         |                                    noisy_z_data.npy",
    "         |",
    "         +--> RayProcessor.run()          --> peaks.json",
    "         |                                    peaks_paired.json",
    "         |",
    "         +--> Sample.organize()           --> files moved to numpy/ and images/",
    "         |",
    "         |--- [for each detected peak (Vx, Vy)]",
    "         |",
    "         |    +--> CroppedRegion          --> *_cropped.npy + grid_mapping.txt",
    "         |    +--> OverlayRenderer        --> ground_truth_labels.npy",
    "         |    +--> Plotter                --> normalized_3d_surface.png",
    "         |    +--> PeakDetector           --> sweep_params_*.txt + *.gif",
    "         |    +--> SummaryWriter          --> summary_local.txt",
    "         |    +--> ChargeSensorBinarizer  --> binary_*.png",
    "         |    +--> CoordinateMapper       --> voltage_coordinates.txt",
    "         |    +--> OverlayRenderer        --> per-peak overlay image",
    "         |",
    "         +--> OverlayRenderer             --> all_rays_peaks_overlay.png",
    "                                              all_scanned_peaks_overlay.png",
    "                                              summary.png",
]
w = pdf.w - pdf.l_margin - pdf.r_margin
for line in diagram:
    pdf.cell(w, 5, line, fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", size=10)
pdf.ln(3)

# ?? SAVE ?????????????????????????????????????????????????????????????????????
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
pdf.output(OUTPUT)
print(f"Report saved to: {OUTPUT}")
