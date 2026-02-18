# Grey-Box Optimization Test Problems for Bi-Level Bayesian Optimization

## Problem Structure Overview

All problems follow the bi-level structure:

**Outer Loop (Bayesian Optimization):**
$$\min_{x_{BB}} J(x_{WB}^*, y)$$
where $y = f_{BB}(x_{BB})$ (expensive black-box evaluation)

**Inner Loop (Nonlinear Programming):**
$$x_{WB}^* = \arg\min_{x_{WB}} J(x_{WB}, y)$$
subject to: $g(x_{WB}, y) \leq 0$ (white-box constraints)

---

## Problem 1: Heat Exchanger with Novel Working Fluid

**Physical Motivation:** Design a heat exchanger system where the working fluid's thermodynamic properties depend on molecular descriptors (black-box), while the heat exchanger geometry and operating conditions are optimized analytically.

### Black-Box Variables ($x_{BB}$)
- $m_1$: Molecular weight proxy ∈ [50, 200]
- $m_2$: Polarity descriptor ∈ [0, 1]

### Black-Box Function
$$y = f_{BB}(x_{BB})$$
- $y_1 = c_p(m_1, m_2)$: Heat capacity [kJ/kg·K]
- $y_2 = \mu(m_1, m_2)$: Viscosity [Pa·s]
- $y_3 = k(m_1, m_2)$: Thermal conductivity [W/m·K]

**Simulated Black-Box (for benchmarking):**
$$y_1 = 2.0 + 0.5\sin\left(\frac{\pi m_1}{50}\right) + 0.3m_2$$
$$y_2 = 0.001 \cdot \exp\left(\frac{m_1 - 100}{50}\right) \cdot (1 + 0.5m_2)$$
$$y_3 = 0.1 + 0.05\left(1 - \left(\frac{m_1 - 125}{75}\right)^2\right) + 0.02m_2$$

### White-Box Variables ($x_{WB}$)
- $A$: Heat transfer area [m²] ∈ [1, 50]
- $\dot{m}$: Mass flow rate [kg/s] ∈ [0.1, 5]
- $\Delta T_{lm}$: Log-mean temperature difference [K] ∈ [5, 50]

### White-Box Equations and Objective
**Objective:** Minimize total cost
$$J = C_{capital} + C_{operating}$$
$$C_{capital} = 1000 \cdot A^{0.6}$$
$$C_{operating} = 500 \cdot \frac{\dot{m}^3 \cdot y_2}{A}$$ (pumping cost scales with viscosity)

**Heat duty constraint:**
$$Q = U \cdot A \cdot \Delta T_{lm} \geq Q_{required} = 100 \text{ kW}$$

**Overall heat transfer coefficient:**
$$U = \frac{y_3}{0.01 + 0.001/y_1}$$

**Reynolds number constraint (turbulent flow):**
$$Re = \frac{4\dot{m}}{\pi \cdot 0.05 \cdot y_2} \geq 4000$$

### Complete Formulation
$$\min_{x_{BB}} J(x_{WB}^*, y) \quad \text{where } y = f_{BB}(x_{BB})$$

**Inner problem:**
$$\min_{A, \dot{m}, \Delta T_{lm}} 1000 \cdot A^{0.6} + 500 \cdot \frac{\dot{m}^3 \cdot y_2}{A}$$
subject to:
- $U \cdot A \cdot \Delta T_{lm} \geq 100$
- $\frac{4\dot{m}}{\pi \cdot 0.05 \cdot y_2} \geq 4000$
- Variable bounds

---

## Problem 2: Pressure Swing Adsorption with Novel Adsorbent

**Physical Motivation:** Design a PSA cycle for gas separation where adsorbent properties come from molecular simulations (black-box), while cycle parameters are optimized using known mass balance equations.

### Black-Box Variables ($x_{BB}$)
- $\sigma$: Pore size parameter [Å] ∈ [3, 10]
- $\epsilon$: Surface energy parameter [kJ/mol] ∈ [5, 25]

### Black-Box Function (Langmuir isotherm parameters)
$$y_1 = q_{max}(\sigma, \epsilon)$$: Maximum loading [mol/kg]
$$y_2 = K_A(\sigma, \epsilon)$$: Adsorption equilibrium constant for component A
$$y_3 = K_B(\sigma, \epsilon)$$: Adsorption equilibrium constant for component B

**Simulated Black-Box:**
$$y_1 = 5.0 \cdot \exp\left(-\frac{(\sigma - 6)^2}{4}\right) \cdot \left(1 + 0.1\epsilon\right)$$
$$y_2 = 0.5 \cdot \exp\left(\frac{\epsilon - 10}{5}\right) \cdot \left(1 - 0.05(\sigma - 5)^2\right)$$
$$y_3 = 0.1 \cdot \exp\left(\frac{\epsilon - 15}{8}\right) \cdot \left(1 + 0.03(\sigma - 7)^2\right)$$

### White-Box Variables ($x_{WB}$)
- $P_H$: High pressure [bar] ∈ [2, 10]
- $P_L$: Low pressure [bar] ∈ [0.1, 1]
- $t_{ads}$: Adsorption time [s] ∈ [10, 120]

### White-Box Equations
**Selectivity:**
$$\alpha = \frac{y_2}{y_3}$$

**Working capacity (simplified):**
$$\Delta q = y_1 \cdot \left(\frac{y_2 \cdot P_H}{1 + y_2 \cdot P_H} - \frac{y_2 \cdot P_L}{1 + y_2 \cdot P_L}\right)$$

**Productivity:**
$$\text{Prod} = \frac{\Delta q}{t_{ads}}$$

**Objective:** Maximize productivity while minimizing energy
$$J = -\text{Prod} + 0.1 \cdot \ln\left(\frac{P_H}{P_L}\right)$$

**Constraints:**
- Minimum selectivity: $\alpha \geq 3$
- Minimum purity (simplified): $\frac{\alpha \cdot P_H}{1 + \alpha \cdot P_H} \geq 0.95$

---

## Problem 3: Batch Reactor with Catalyst Optimization

**Physical Motivation:** Optimize a batch reactor operation where catalyst properties affect reaction kinetics (black-box from DFT calculations), while reactor conditions follow known mass/energy balances.

### Black-Box Variables ($x_{BB}$)
- $E_b$: Binding energy [eV] ∈ [-2, 0]
- $d$: Active site density proxy ∈ [0.5, 2]

### Black-Box Function (Kinetic parameters)
$$y_1 = k_1(E_b, d)$$: Forward rate constant [1/s]
$$y_2 = k_2(E_b, d)$$: Side reaction rate constant [1/s]
$$y_3 = K_{eq}(E_b, d)$$: Equilibrium constant [-]

**Simulated Black-Box (Sabatier-type volcano relationship):**
$$y_1 = 10 \cdot d \cdot \exp\left(-\frac{(E_b + 1)^2}{0.25}\right)$$
$$y_2 = 0.5 \cdot d \cdot \exp\left(-\frac{(E_b + 0.5)^2}{0.5}\right)$$
$$y_3 = \exp\left(-2 \cdot E_b - 1\right)$$

### White-Box Variables ($x_{WB}$)
- $T$: Temperature [K] ∈ [300, 500]
- $t_f$: Final reaction time [hr] ∈ [0.5, 10]
- $C_{A,0}$: Initial concentration [mol/L] ∈ [0.5, 5]

### White-Box Equations (Batch reactor)
**Arrhenius-modified rates:**
$$k_1' = y_1 \cdot \exp\left(\frac{-5000}{T}\right)$$
$$k_2' = y_2 \cdot \exp\left(\frac{-6000}{T}\right)$$

**Concentrations at final time (analytical solution for A→B→C):**
$$C_A = C_{A,0} \cdot \exp(-k_1' \cdot t_f)$$
$$C_B = C_{A,0} \cdot \frac{k_1'}{k_2' - k_1'} \cdot \left(\exp(-k_1' \cdot t_f) - \exp(-k_2' \cdot t_f)\right)$$

**Objective:** Maximize yield of B minus operating costs
$$J = -C_B + 0.001 \cdot T \cdot t_f$$

**Constraints:**
- Minimum conversion: $1 - C_A/C_{A,0} \geq 0.8$
- Selectivity: $C_B/(C_{A,0} - C_A) \geq 0.7$

---

## Problem 4: Distillation Column with Novel Solvent (Extractive Distillation)

**Physical Motivation:** Design an extractive distillation column where solvent properties come from COSMO-RS predictions (black-box), while column design follows standard McCabe-Thiele equations.

### Black-Box Variables ($x_{BB}$)
- $\delta_H$: Hansen solubility parameter (hydrogen bonding) ∈ [5, 25]
- $\delta_P$: Hansen solubility parameter (polar) ∈ [5, 20]

### Black-Box Function (VLE modification)
$$y_1 = \alpha_{12}^{mod}(\delta_H, \delta_P)$$: Modified relative volatility
$$y_2 = \rho(\delta_H, \delta_P)$$: Solvent density [kg/m³]
$$y_3 = \mu(\delta_H, \delta_P)$$: Solvent viscosity [cP]

**Simulated Black-Box:**
$$y_1 = 1.5 + 2.0 \cdot \sin\left(\frac{\pi(\delta_H - 10)}{20}\right) \cdot \cos\left(\frac{\pi(\delta_P - 10)}{15}\right)$$
$$y_2 = 800 + 50 \cdot \delta_H - 20 \cdot \delta_P$$
$$y_3 = 0.5 + 0.1 \cdot \delta_H + 0.05 \cdot \delta_P^2 / 100$$

### White-Box Variables ($x_{WB}$)
- $R$: Reflux ratio ∈ [1, 10]
- $N$: Number of stages ∈ [5, 50] (treated as continuous for NLP)
- $S/F$: Solvent-to-feed ratio ∈ [0.5, 5]

### White-Box Equations
**Minimum reflux (Underwood):**
$$R_{min} = \frac{1}{y_1 - 1} \cdot \left(\frac{x_D}{x_F} - y_1 \cdot \frac{1 - x_D}{1 - x_F}\right)$$
where $x_D = 0.99$ (distillate purity), $x_F = 0.5$ (feed composition)

**Minimum stages (Fenske):**
$$N_{min} = \frac{\ln\left(\frac{x_D(1-x_B)}{x_B(1-x_D)}\right)}{\ln(y_1)}$$
where $x_B = 0.01$ (bottoms impurity)

**Gilliland correlation:**
$$\frac{N - N_{min}}{N + 1} = 0.75 \cdot \left(1 - \left(\frac{R - R_{min}}{R + 1}\right)^{0.566}\right)$$

**Objective:** Minimize total annual cost
$$J = 1000 \cdot N + 500 \cdot R + 200 \cdot (S/F) \cdot y_3$$

**Constraints:**
- $R \geq 1.2 \cdot R_{min}$
- $N \geq 1.5 \cdot N_{min}$
- Solvent recovery constraint: $(S/F) \cdot y_2 \leq 5000$

---

## Problem 5: Multi-Effect Evaporator with Phase-Change Material

**Physical Motivation:** Design a multi-effect evaporator system where the heat transfer fluid properties come from molecular design (black-box), while the evaporator network follows known energy balances.

### Black-Box Variables ($x_{BB}$)
- $T_m$: Melting point proxy ∈ [50, 150]
- $\lambda$: Latent heat scaling ∈ [0.5, 2]

### Black-Box Function
$$y_1 = \Delta H_{fus}(T_m, \lambda)$$: Latent heat of fusion [kJ/kg]
$$y_2 = c_p^{liq}(T_m, \lambda)$$: Liquid heat capacity [kJ/kg·K]
$$y_3 = T_{melt}(T_m, \lambda)$$: Actual melting temperature [°C]

**Simulated Black-Box:**
$$y_1 = 150 \cdot \lambda \cdot \left(1 + 0.2 \sin\left(\frac{\pi T_m}{100}\right)\right)$$
$$y_2 = 2.0 + 0.5 \cdot \lambda - 0.002 \cdot T_m$$
$$y_3 = T_m + 10 \cdot \sin\left(\frac{\pi \lambda}{2}\right)$$

### White-Box Variables ($x_{WB}$)
- $n$: Number of effects ∈ [2, 6]
- $\dot{m}_{HTF}$: HTF mass flow rate [kg/s] ∈ [1, 20]
- $\Delta T_{effect}$: Temperature drop per effect [K] ∈ [5, 20]

### White-Box Equations
**Heat available from HTF:**
$$Q_{HTF} = \dot{m}_{HTF} \cdot (y_1 + y_2 \cdot 10)$$ (latent + sensible)

**Evaporation rate:**
$$\dot{m}_{evap} = \frac{Q_{HTF} \cdot n \cdot 0.9^{n-1}}{2260}$$ (decreasing efficiency)

**Objective:** Maximize evaporation rate minus costs
$$J = -\dot{m}_{evap} + 0.1 \cdot n^2 + 0.01 \cdot \dot{m}_{HTF}^2$$

**Constraints:**
- Temperature feasibility: $y_3 > 100 + n \cdot \Delta T_{effect}$
- Minimum evaporation: $\dot{m}_{evap} \geq 5$ kg/s

---

## Problem 6: Membrane Separation with Designed Polymer

**Physical Motivation:** Design a membrane separation process where membrane permeability/selectivity come from polymer structure (black-box), while the membrane module design follows known transport equations.

### Black-Box Variables ($x_{BB}$)
- $FFV$: Fractional free volume ∈ [0.1, 0.3]
- $d_{spacing}$: d-spacing parameter [Å] ∈ [3, 8]

### Black-Box Function (following Robeson upper bound concepts)
$$y_1 = P_A(FFV, d_{spacing})$$: Permeability of component A [Barrer]
$$y_2 = \alpha_{A/B}(FFV, d_{spacing})$$: Selectivity A over B
$$y_3 = \sigma(FFV, d_{spacing})$$: Mechanical strength proxy [MPa]

**Simulated Black-Box (trade-off relationship):**
$$y_1 = 1000 \cdot \exp\left(\frac{FFV - 0.15}{0.05}\right) \cdot \left(1 + 0.1 \cdot d_{spacing}\right)$$
$$y_2 = 50 \cdot \exp\left(-\frac{FFV - 0.15}{0.1}\right) \cdot \exp\left(-\frac{(d_{spacing} - 5)^2}{4}\right)$$
$$y_3 = 100 \cdot (0.4 - FFV) \cdot (10 - d_{spacing})$$

### White-Box Variables ($x_{WB}$)
- $A_m$: Membrane area [m²] ∈ [10, 1000]
- $\delta$: Membrane thickness [μm] ∈ [0.1, 10]
- $\Delta p$: Pressure difference [bar] ∈ [1, 20]

### White-Box Equations
**Flux through membrane:**
$$J_A = \frac{y_1 \cdot \Delta p}{\delta} \cdot 10^{-10}$$ (unit conversion)

**Permeate flow:**
$$F_A = J_A \cdot A_m$$

**Objective:** Maximize productivity minus costs
$$J = -F_A \cdot y_2 + 0.1 \cdot A_m + 10 \cdot \Delta p$$

**Constraints:**
- Minimum flux: $J_A \geq 10^{-6}$
- Mechanical stability: $\Delta p \cdot A_m / y_3 \leq 100$
- Minimum selectivity in permeate: product purity constraint

---

## Summary Table

| Problem | $n_{BB}$ | $n_y$ | $n_{WB}$ | $n_g$ | Domain |
|---------|----------|-------|----------|-------|--------|
| 1. Heat Exchanger | 2 | 3 | 3 | 2 | Heat transfer |
| 2. PSA | 2 | 3 | 3 | 2 | Separation |
| 3. Batch Reactor | 2 | 3 | 3 | 2 | Reaction |
| 4. Distillation | 2 | 3 | 3 | 3 | Separation |
| 5. Evaporator | 2 | 3 | 3 | 2 | Heat transfer |
| 6. Membrane | 2 | 3 | 3 | 2 | Separation |

All problems feature:
- 2 black-box decision variables (low-dimensional for BO efficiency)
- 3 black-box outputs (intermediate parameters)
- 3 white-box decision variables (typical for process design)
- 2-3 inequality constraints (meaningful engineering limits)
- Nonlinear coupling between black-box outputs and white-box optimization
- Physically meaningful interpretations
