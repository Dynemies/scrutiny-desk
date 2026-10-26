"""FFDN (Chen et al., ECCV 2024), as implemented in ForensicHub (Sichuan University, CC BY 4.0,
https://github.com/scu-zjz/ForensicHub). Copied with two changes: no ForensicHub registry/base-class imports,
and the ImageNet backbone file is only loaded when a path is given (our checkpoint already holds all weights)."""
