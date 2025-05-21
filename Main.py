from Sim import Sim
import warnings
import pandas as pd
import numpy as np
# warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)

warnings.filterwarnings("ignore")

# warnings.simplefilter("always")  # Always show full warning stack
Sim(Scenario='Scenario3 London Ver3',Build_White_Picture='Rebuild',Build_Plots=True,Build_Tracks=True,BuildVisualization=True,ShowTracksErrors=True,ShowPlotsErrors=True)
input("Press Enter to terminate...")
