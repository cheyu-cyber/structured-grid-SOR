=== uname
Linux scc-j06 4.18.0-553.94.1.el8_10.x86_64 #1 SMP Mon Jan 19 05:45:41 EST 2026 x86_64 x86_64 x86_64 GNU/Linux
=== nproc (cgroup-limited in this session)
1
=== lscpu (actual hardware)
Architecture:        x86_64
CPU(s):              32
Thread(s) per core:  1
Core(s) per socket:  16
Socket(s):           2
NUMA node(s):        2
Model name:          Intel(R) Xeon(R) Gold 6426Y
L1d cache:           48K
L1i cache:           32K
L2 cache:            2048K
L3 cache:            38400K
NUMA node0 CPU(s):   0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
NUMA node1 CPU(s):   1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
=== /proc/cpuinfo cache
cache size	: 38400 KB
=== free -h
              total        used        free      shared  buff/cache   available
Mem:          250Gi       4.0Gi       125Gi        41Mi       121Gi       245Gi
Swap:         8.0Gi        64Mi       7.9Gi
=== nvidia-smi
Sat Apr 25 21:21:10 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40S                    On  |   00000000:49:00.0 Off |                    0 |
| N/A   54C    P0            191W /  350W |   10077MiB /  46068MiB |     93%   E. Process |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L40S                    On  |   00000000:61:00.0 Off |                    0 |
| N/A   32C    P8             32W /  350W |       0MiB /  46068MiB |      0%   E. Process |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA L40S                    On  |   00000000:C9:00.0 Off |                    0 |
| N/A   33C    P8             33W /  350W |       0MiB /  46068MiB |      0%   E. Process |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA L40S                    On  |   00000000:E1:00.0 Off |                    0 |
| N/A   32C    P8             33W /  350W |       0MiB /  46068MiB |      0%   E. Process |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2862029      C   python                                10068MiB |
+-----------------------------------------------------------------------------------------+
=== nvcc --version
/bin/bash: line 28: nvcc: command not found
=== Python venv
Python 3.13.8
pandas 3.0.2 / matplotlib 3.10.9 / numpy 2.4.4
