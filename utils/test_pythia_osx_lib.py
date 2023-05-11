

import ctypes

# Load the Pythia library
lib = ctypes.CDLL("/usr/local/lib/pythia8309/lib/libpythia8.dylib")

# Create a Pythia instance
pythia = lib.Pythia8()

# Set up the Pythia configuration
pythia.readString("Beams:eCM = 13000.")  # center-of-mass energy
pythia.readString("HardQCD:all = on")  # turn on hard QCD processes
pythia.readString("PhaseSpace:pTHatMin = 20.")  # minimum pT
pythia.readString("Random:setSeed = on")  # turn on random seeding
pythia.readString("Random:seed = 0")  # set random seed to 0

# set radius and length of cylindrical detector
pythia.readString("Beams:frameType = 1")
pythia.readString("Detector:UseCMSSW = off")
pythia.readString("Detector:Rmax = 2.0")
pythia.readString("Detector:Zmax = 5.0")

# Initialize the Pythia instance
pythia.init()

# Generate events
for i in range(10):
    if not pythia.next(): continue
    event = pythia.event
    print("Event ", i, " with ", event.size(), " particles")

# End Pythia
pythia.stat()
