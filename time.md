## 本地：GTX1650

|                        | 未优化      | 优化       |
| ---------------------- |:--------:|:--------:|
| backprop               | 3.1534ms | 1.9571ms |
| cfd                    | 1142ms   | 1136ms   |
| cfd double             | 6896ms   | 6982ms   |
| comd                   | 326ms    | 319ms    |
| lbm                    | 826ms    | 643ms    |
| lulesh                 | 2289ms   | 2382ms   |
| mix11                  | 878ms    | 884ms    |
| particlefilter         | 217ms    | 217ms    |
| particlefilter naive   | 10.58ms  | 10.68ms  |
| PP_FP_MEM              | 1.093ms  | 1.094ms  |
| sad                    | 11.321ms | 10.01ms  |
| sdk-matrixMul-modified | 275ms    | 275ms    |
| srad_v1                | 22.47ms  | 21.27ms  |

## A100

|                        | 未优化       | 优化        |
| ---------------------- |:---------:|:---------:|
| backprop               | 0.1261ms  | 0.1245ms  |
| cfd                    | -(4299ms) | -(4288ms) |
| cfd double             | -         | -         |
| comd                   | 108ms     | 107ms     |
| lbm                    | 79ms      | 57ms      |
| lulesh                 | 1022ms    | 1028ms    |
| mix11                  | 180ms     | 182ms     |
| particlefilter         | 39ms      | 42ms      |
| particlefilter naive   | 10.58ms   | 10.68ms   |
| PP_FP_MEM              | 0.66ms    | 1.13ms    |
| sad                    | 1.127ms   | 1.05ms    |
| sdk-matrixMul-modified | 42ms/47ms | 47ms      |
| srad_v1                | 6.28ms    | 5.84ms    |
