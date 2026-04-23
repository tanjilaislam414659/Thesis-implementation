# ColPack Command Template

## General template

```bat
test_colpack.exe "FULL_PATH_TO_MATRIX.mtx" ORDERING_NAME > output_file.txt
```

## Purpose

This command runs the ColPack test runner on a matrix file with a chosen ordering strategy and saves the full output to a text file.

## Current examples

### ash85 with SMALLEST_LAST

```bat
test_colpack.exe "C:\Users\riyat\Documents\Thesis Implementations\data\raw\matrices\ash85.mtx" SMALLEST_LAST > ash85_smallest_last.txt
```

### ash85 with LARGEST_FIRST

```bat
test_colpack.exe "C:\Users\riyat\Documents\Thesis Implementations\data\raw\matrices\ash85.mtx" LARGEST_FIRST > ash85_largest_first.txt
```
Observed result:
- SMALLEST_LAST -> 4 colors
- LARGEST_FIRST -> 5 colors


### can_24 with SMALLEST_LAST

```bat
test_colpack.exe "C:\Users\riyat\Documents\Thesis Implementations\data\raw\matrices\can_24.mtx" SMALLEST_LAST > can24_smallest_last.txt
```

### can_24 with LARGEST_FIRST

```bat
test_colpack.exe "C:\Users\riyat\Documents\Thesis Implementations\data\raw\matrices\can_24.mtx" LARGEST_FIRST > can24_largest_first.txt
```
Observed result:
- SMALLEST_LAST -> 4 colors
- LARGEST_FIRST -> 5 colors


### jac_pat with SMALLEST_LAST

```bat
test_colpack.exe "C:\Users\riyat\Documents\Thesis Implementations\data\raw\matrices\jac_pat.mtx" SMALLEST_LAST > jacpat_smallest_last.txt
```

### jac_pat with LARGEST_FIRST

```bat
test_colpack.exe "C:\Users\riyat\Documents\Thesis Implementations\data\raw\matrices\jac_pat.mtx" LARGEST_FIRST > jacpat_largest_first.txt
```

Observed result:
- SMALLEST_LAST -> 5 colors
- LARGEST_FIRST -> 5 colors

## Notes

- The first argument is the full path to the matrix file.
- The second argument is the ColPack ordering name.
- The output is redirected into a text file for later inspection and parsing.
- The current runner prints the requested ordering, coloring distance, input graph, summary metrics, and per-vertex colors.


## Initial baseline orderings

For the current ColPack baseline workflow, the initial orderings used are:

- SMALLEST_LAST
- LARGEST_FIRST

These two orderings are used first because they already run successfully in the current runner workflow and provide a simple but meaningful baseline comparison across matrices.
