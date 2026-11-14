# Weight-Space Programming vs Neural ODE
The November 2, 2026 article, Sparks of Weight-Space Programming, by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab at MIT CSAIL, reports three capabilities. 
rikiyat.github.io

## Read weights

Given raw neural-network parameters, the LLM can sometimes infer what the network computes without executing it. In one experiment, it identified which game a small policy network had learned.
## Write weights

The LLM constructs network parameters directly from a description of desired behavior. In experiments on small game-playing networks, these generated policies outperformed selected PPO-trained baselines.


## Repair weights

The LLM identifies a behavioral defect and modifies a small subset of parameters. One reported experiment substantially reduced reasoning loops in a small language model.

# Research Direction 
Weight-Space Programming (WSP) could be an interesting alternative to conventional training for Physics-Informed Neural Networks (PINNs), Neural ODEs, and neural operators.
