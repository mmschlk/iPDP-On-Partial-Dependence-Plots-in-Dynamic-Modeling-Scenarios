# Paper Supplement: iPDP: On Partial Dependence Plots in Dynamic Modeling Scenarios

The experiments and results presented in the paper are available in the `projects/iPDP` module.

The `projects/iPDP` module contains the following submodules:

- `batch_california.py`: Script to run the experiments on the California Housing dataset.
- `change_detector_elec2.py`: Script to run the experiments on the Elec2 data stream.
- `agrawal_concept_drifts.py`: Create and run the experiment on the Agrawal data stream.
- `change_detector.py`: Module containing the change detector class.
- `rotating_plane.py`: Script to run the experiments on the Rotating Hyperplane data stream.

The implementation of the iPDP can be found in the `iXAI` module. 
Specifically, the `iXAI.ixai.explainers.pdp` module contains the implementation of the iPDP.
The `iXAI.ixai.utils.tracker.extreme_value_tracker` module contains the implementation of the maximum/minimum value tracker.
The `iXAI/ixai/storage/ordered_reservoir_storage` contains the extended reservoir storage implementation for frequency storage mechanisms.

