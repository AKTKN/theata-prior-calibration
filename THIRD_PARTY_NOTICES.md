# Third-party notices

This project uses [THeatA](https://github.com/MSTakahashiPJ/theata) at commit
`8bfb9b97de91bacac2f56725c27b4c71c6a91009`, distributed under the Apache License
2.0. The upstream checkout is an external dependency and is not redistributed
in this package archive.

`src/theata_prior/circuits.py` includes a modified implementation of THeatA's
ASAP scheduler grouping rule. The modification builds timestep buckets in one
pass instead of rescanning all operations per timestep. Zone constraints,
earliest-step selection, operation order within each step, and compiled
semantics are retained and tested against upstream. A copy of the applicable
license is included in `licenses/THEATA-APACHE-2.0.txt`.

Stim, PyMatching, NumPy, SciPy and the other dependencies retain their own
licenses and are installed separately.
