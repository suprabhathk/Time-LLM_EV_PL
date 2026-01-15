# Age-heterogeneous SIR model



For each age group $i$, out of a set of $n$ disjoint age groups, we have:

$$\begin{aligned}
\dot{S}_i(t) &= -\beta S_i(t) \sum_{j=0}^{n} M_{ij} \frac{I_j(t)}{N_j(t)} \\
\dot{I}_i(t) &= \beta S_i(t) \sum_{j=0}^{n} M_{ij} \frac{I_j(t)}{N_j(t)} - \gamma I_i(t) \\
\dot{R}_i(t) &= \gamma I_i(t),
\end{aligned} \tag{3.11}$$

where **$\beta$** is the probability of infection when a contact takes place, **$\gamma$** is the recovery rate and **$M_{ij}$** is the average frequency of contacts that an individual in age group $i$ has with an individual in age group $j$, with,

$$S_i(0) > 0, E_i(0) > 0, I_i(0) \geq 0, \text{ and } R_i(0) = 0,

and the total population $N_i$ is constant:

$$N_i(t) = S_i(t) + E_i(t) + I_i(t) + R_i(t)

---

### 🔢 Population Census Data and Contact Matrix

The population census data, indexed with **C** for children and **A** for adults, is:

$$\begin{aligned}
N_C &= 1500000 \\
N_A &= 3500000 \\
S_C(0) &= N_C - 1 \\
S_A(0) &= N_A - 1 \\
I_C(0) &= I_A(0) = 1,
\end{aligned} 

and the contact matrix:

$$M = \begin{array}{c} C \\ A \end{array} \begin{pmatrix} C & A \\ 18 & 7 \\ 3 & 12 \end{pmatrix}

Note that the contact matrix $M$ should be reciprocal, such that we have:

$$N_i M_{ij} = N_j M_{ji}
