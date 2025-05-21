import os
import tkinter as tk
from tkinter import ttk
import Sim


def get_scenarios(folder='C:\\Drones\\Python\\Scenarios'):
    """Get list of subfolders under the given folder."""
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def run_sim():
    """Run the Sim function with selected parameters."""
    scenario = scenario_var.get()
    if not scenario:
        return
    build_white_picture = build_white_var.get()
    build_plots = build_plots_var.get()
    build_tracks = build_tracks_var.get()
    build_visualization = build_visualization_var.get()
    show_tracks_errors = show_tracks_errors_var.get()
    show_plots_errors = show_plots_errors_var.get()


    # Call Sim function with selected parameters

    Sim.Sim(
        Scenario=scenario,
        Build_White_Picture=build_white_picture,
        Build_Plots=build_plots,
        Build_Tracks=build_tracks,
        BuildVisualization=build_visualization,
        ShowTracksErrors=show_tracks_errors,
        ShowPlotsErrors=show_plots_errors
        )


def toggle_controls(*args):
    """Enable or disable controls based on whether a scenario is selected."""
    state = "normal" if scenario_var.get() else "disabled"
    build_white_dropdown.config(state=state)
    build_plots_check.config(state=state)

    build_tracks_check.config(state=state)
    build_visualization_check.config(state=state)
    show_tracks_errors_check.config(state=state)
    show_plots_errors_check.config(state=state)
    run_button.config(state=state)


# Create GUI window
root = tk.Tk()
root.title("Sim Runner")

# Scenario selection
tk.Label(root, text="Select Scenario:").grid(row=0, column=0)
scenario_var = tk.StringVar()
scenario_var.trace_add("write", toggle_controls)
scenarios = get_scenarios()
scenario_dropdown = ttk.Combobox(root, textvariable=scenario_var, values=scenarios, state="readonly")
scenario_dropdown.grid(row=0, column=1)

# Build White Picture selection
tk.Label(root, text="Build White Picture:").grid(row=1, column=0)
build_white_var = tk.StringVar(value="Rebuild")
build_white_dropdown = ttk.Combobox(root, textvariable=build_white_var,
                                    values=["Rebuild", "Load Matlab", "Load Python","External CSV"], state="disabled")
build_white_dropdown.grid(row=1, column=1)

# Boolean checkboxes
build_plots_var = tk.BooleanVar(value=True)
build_plots_check = tk.Checkbutton(root, text="Build Plots", variable=build_plots_var, state="disabled")
build_plots_check.grid(row=2, column=0, sticky="w")

build_tracks_var = tk.BooleanVar(value=True)
build_tracks_check = tk.Checkbutton(root, text="Build Tracks", variable=build_tracks_var, state="disabled")
build_tracks_check.grid(row=3, column=0, sticky="w")

build_visualization_var = tk.BooleanVar(value=True)
build_visualization_check = tk.Checkbutton(root, text="Build Visualization", variable=build_visualization_var,
                                           state="disabled")
build_visualization_check.grid(row=4, column=0, sticky="w")

show_tracks_errors_var = tk.BooleanVar(value=True)
show_tracks_errors_check = tk.Checkbutton(root, text="Show Tracks Errors", variable=show_tracks_errors_var,
                                          state="disabled")
show_tracks_errors_check.grid(row=5, column=0, sticky="w")

show_plots_errors_var = tk.BooleanVar(value=True)
show_plots_errors_check = tk.Checkbutton(root, text="Show Plots Errors", variable=show_plots_errors_var,
                                         state="disabled")
show_plots_errors_check.grid(row=6, column=0, sticky="w")

# Boolean checkboxes

# Run button
run_button = tk.Button(root, text="Run Sim", command=run_sim, state="disabled")
run_button.grid(row=8, column=0, columnspan=2)

# Start GUI loop
root.mainloop()
