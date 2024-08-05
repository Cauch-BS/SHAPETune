# Black Box Optimization

*Black-Box optimization* refers to the optimization of an objective function whose internal structure is not known or not accessible. 

The primary challenge in black-box optimization is to enhance the quality of the objective function value with minimal amount of calls to that objective function. 

In our case, we are aiming to optimize the partition function to match the SHAPE reactivity data. A partition function is a prototypical example of a black box function. Due to the complexity of the RNA sequence, understanding the RNA partition function for a long RNA sequence is practically intractable. 

One should note that calculating the partition function is no mean feat. The fast algorithm available, which is LinearPartition runs in $\mathcal{O}(n)$ time. Running on a `64` core AMD EPYC cpu, calculating the partition function for a luciferase sequence `1650`bp long `760` times takes on average `100` seconds, making the task of optimization non-trivial. 