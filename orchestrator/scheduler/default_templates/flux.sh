#!/bin/bash
#flux: -N <NODES>
#flux: -q <QUEUE>
#flux: -B <ACCOUNT>
#flux: -t <WALLTIME>
#flux: --exclusive
<EXTRA_HEADER>

<PREAMBLE>

<COMMAND>

<POSTAMBLE>

touch job_done
