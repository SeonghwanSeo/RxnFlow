# Protein used in experiments

## LIT-PCBA optimization

From https://drugdesign.unistra.fr/LIT-PCBA/

| Target     | PDB ID | Center                |
| ---------- | ------ | --------------------- |
| ADRB2      | 4ldo   | -1.96, -12.27, -48.98 |
| ALDH1      | 5l2m   | 34.43, -16.88, 13.77  |
| ESR_ago    | 2p15   | -35.22, 4.64, 20.78   |
| ESR_antago | 2iok   | 17.85, 35.51, 52.49   |
| FEN1       | 5fv7   | -16.81, -4.80, 0.62   |
| GBA        | 2v3d   | 32.44, 33.88, -19.56  |
| IDH1       | 4umx   | 12.11, 28.09, 80.47   |
| KAT2A      | 5h86   | -0.11, 5.73, 10.14    |
| MAPK1      | 4zzn   | -15.69, 14.49, 42.72  |
| MTORC1     | 4dri   | 35.38, 49.65, 36.21   |
| OPRK1      | 6b73   | 58.61, -24.16, -4.32  |
| PKM2       | 4jpg   | 8.64, 2.94, 10.76     |
| PPARG      | 5y2t   | 8.30, -1.02, 46.32    |
| TP53       | 3zme   | 89.32, 91.82, -44.87  |
| VDR        | 3a2i   | 11.38, -3.12, -31.57  |

### Note

There is some valence issue, we remove 2623'th Atom (O) in PPARG.

## SBDD optimization (zero-shot sampling with pocket-conditioning)

From https://github.com/gnina/models/tree/master/data/CrossDocked2020
15,201 training pockets + 100 test pockets.
