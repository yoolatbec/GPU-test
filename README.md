original目录下为从原始仓库中克隆得到的文件，modified目录下为修改后得到的文件，同时在子项目录中已经包含了运行所需要的数据文件。original中包含的仓库有：

* gpu-app-collection，仓库地址：<a href="https://github.com/accel-sim/gpu-app-collection">https://github.com/accel-sim/gpu-app-collection</a>

修改的目的是使GPU函数中连续的读写操作变得分散。修改包括：

* 将连续的读写分开使得两个读写操作之间间隔若干非读写操作，如lbm, sad, srad_v1

* 将类似+=的运算分为一个算术运算和一个赋值，并将读写操作集中起来

不保证修改前后函数的等价性。如果一个函数的命名为xxx_seperate，那么这个函数中存在部分读写操作被分散，并且存在对应的函数xxx，该部分读写没有被分散。编译时通过定义宏**SEPERATE** 确定使用的函数。如果希望使用读写被分散的函数，在编译时加入**\-DSEPERATE**选项

使用cudaEvent计时。

## 编译运行

进入到对应的子项目录

```
make run
```

如果需要改变两个版本的程序的运行顺序：

```
make run_inv
```

## 运行时间

### 本地：GTX1650+nvcc

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

### A100+nvcc

|                        | 未优化       | 优化        |
| ---------------------- |:---------:|:---------:|
| backprop               | 0.1261ms  | 0.1245ms  |
| cfd*                   | -(4299ms) | -(4288ms) |
| cfd double             | -         | -         |
| comd                   | 108ms     | 107ms     |
| lbm                    | 79ms      | 57ms      |
| lulesh                 | 1022ms    | 1028ms    |
| mix11                  | 180ms     | 182ms     |
| particlefilter         | 39ms      | 39ms      |
| particlefilter naive   | 10.58ms   | 10.68ms   |
| PP_FP_MEM              | 0.66ms    | 0.66ms    |
| sad                    | 1.127ms   | 1.05ms    |
| sdk-matrixMul-modified | 42ms/47ms | 47ms      |
| srad_v1                | 6.28ms    | 5.84ms    |

\* cfd和cfd double在A100上没有正常运行。
