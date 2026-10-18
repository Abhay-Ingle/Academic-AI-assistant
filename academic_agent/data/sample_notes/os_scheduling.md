# Operating Systems - CPU Scheduling

CPU scheduling decides which process in the ready queue gets the CPU next. The goals are high CPU utilisation, high throughput, low turnaround time, low waiting time and low response time.

## First Come First Served (FCFS)
Processes are executed in the order they arrive. It is simple but suffers from the convoy effect: a long process at the front makes short processes wait. FCFS is non-preemptive.

## Shortest Job First (SJF)
The process with the smallest next CPU burst runs first. SJF gives the minimum average waiting time for a given set of processes, but the burst length must be predicted and long processes may starve. The preemptive version is called Shortest Remaining Time First (SRTF).

## Round Robin (RR)
Each process receives a fixed time quantum (for example 4 ms) and is then preempted and moved to the back of the ready queue. Round Robin is fair and gives good response time. If the quantum is too large it behaves like FCFS; if it is too small, context-switch overhead dominates.

## Priority Scheduling
The CPU is given to the process with the highest priority. Starvation of low-priority processes can be solved with aging, which gradually increases the priority of waiting processes.

## Deadlock
A deadlock needs four conditions at the same time: mutual exclusion, hold and wait, no preemption and circular wait. Deadlocks can be prevented by breaking one condition, avoided using the Banker's algorithm, or detected and recovered from.
