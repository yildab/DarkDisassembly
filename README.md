# DarkDisassembly: DM Disassembly in the Earth

This code simulates the trajectory of loosely bound composite DM that is 
disassembled into individual constituents as it traverses the Earth.

The file `get_trajectories.py` simulates constituent trajectories for a range
of constituent masses and cross-sections with SM nucleons in the Earth. It
outputs the position and velocities of each constituent at each of its scatters
in the Earth, and at its exit point.

The file `get_spreadinfo.py` uses the final location and velocity output to 
generate various characteristics of the spread, such as its size.

The file `get_timing.py` calculates the expected time difference between
successive scatters in a detector due to a cloud of constituents traversing
the detector.
