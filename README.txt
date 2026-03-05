# Conway's Game of Life and SIRS model

Author: Thomas Horn, s2163159@ed.ac.uk




#### Conway's Game of Life

## Overview

  life.py is used to generate a 2D periodic lattice of 'alive' and 'dead' cells that then updates following the deterministic rules of Conway's game of life.
  It can be used to visualise the progression, or measure how long it takes for the lattice to reach an equilibrium state.
  The lattice can be initialised randomly or from a set of predetermined initial states

## Dependencies

    numpy
    scipy
    matplotlib

## Output

  Time taken for a large number of randomly initialised lattices to reach equilibrium states:
    'equilibration times.csv' / 'equilibration time.png'

## Usage

  >> python life.py [-h] [-a ACTION] [-l LENGTH] [-r RUNS] [-s STATE]

## Options

  -h, --help           show this help message and exit
  -a, --action ACTION  What to do with model: 'animate' (animation) or 'equilibrate' (calculate equilibration time) or 'graph' (plot equilibration graph,
                       requires .csv files from 'measure') or 'glider_speed' (calculate the CoM velocity of a glider). Default='animate'
  -l, --length LENGTH  LxL size of lattice. Default=50
  -r, --runs RUNS      Number of runs when calculating equilibration time. Default=10000
  -s, --state STATE    Initialise the system with a specific state (eg. 'Glider')

## States

  When initialising the lattice with a --state, these are the available options:
    glider
    toad
    pulsar
    gosper_glider_gun




#### SIRS Model

## Overview

  sirs.py simulates the stochastic SIRS (Susceptible–Infected–Recovered–Susceptible) model on a periodic 2D lattice.
  The model can be used to animate the disease spread, generate a phase diagram of the steady-state infected population,
  measure the variance of infection near the phase transition, and plot the quenching effect of permanent immunity.

## Dependencies

    numpy
    numba
    matplotlib
    pandas

## Output

    Phase diagram of the average infected fraction:
      'means.csv' / 'means.png'

    Variance of the infected population along a parameter cut:
      'variances.csv' / 'variances.png'

    Effect of permanent immunity on infection prevalence:
      'immunity.csv' / 'immunity.png'

## Usage

  >> python sirs.py [-h] [-a ACTION] [-L LENGTH] [-s SWEEPS] [-pi PI] [-pr PR] [-ps PS] [--immune IMMUNE]

## Options

  -h, --help           show this help message and exit
  -a, --action ACTION  What to do with model: 'animate' (animation) or 'phase' (calculate phases) or 'graph_phase' (create equilibration graph without
                       recalculating; requires .csv files from 'phase') or 'variance' (calculates var(I/N) along P(s->i)=0.2-0.5 while keeping
                       P(i->r)=P(r->s)=0.5) or 'graph_variance' (create variance graph without recalculating; requires .csv files from 'phase').
                       Default='animate'
  -L, --length LENGTH  LxL size of lattice. Default=50
  -s, --sweeps SWEEPS  Number of sweeps per measurement.
  -psi PI               Probability of infection p(s->i)
  -pir PR               Probability of recovery p(i->r)
  -prs PS               Probability of becoming susceptible p(r->s)
  --immune IMMUNE      Percentage of the inital population given full immunity. Default=0

## Notable Input Parameters

  -psi 0.1 -pir 0.9 -prs 0.9    Infection is eradicated (all states become susceptible)
  -psi 0.5 -pir 0.5 -prs 0.5    Dynamic equilibrium between all 3 states
  -psi 0.8 -pir 0.4 -prs 0.07   System alternates between highly infected and highly susceptible
