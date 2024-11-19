# Project Numerisis

Algorithms for scientific computing implemented in `rust` with
parallelization support via `rayon`.

## Current Features

### `ndiff` Module

Algorithms for numerical differentiation. Currently support backward and central differentiation scheme. 

- Support Parallelization - Partial

### `pde` Module

Algorithms for numerical PDE. Currently support first order-PDEs solution in explicit-Euler algorithm. 
Can be combined with `ndiff`.

- Support Parallelization - No