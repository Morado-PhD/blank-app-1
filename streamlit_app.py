import streamlit as st
import numpy as np
from scipy.optimize import fsolve

# ---------- Functions for the Initial Guess Calculations ----------

def Liq_Vapor_PressureCO2(AV, BV, CV, DV, EV, T):
    return np.exp(AV + (BV / T) + CV * np.log(T) + DV * T**EV)

def Liq_Vapor_PressureCS2(Av, Bv, Cv, Dv, Ev, T):
    return np.exp(Av + (Bv / T) + Cv * np.log(T) + Dv * T**Ev)

def L_rhoCO2(A, B, C, D, T):
    # Returns liquid density in mol/m³ (after unit conversion)
    return (A / B**(1 + (1 - (T / C))**D)) * 1000

def L_rhoCS2(AL, BL, CL, DL, T):
    return (AL / BL**(1 + (1 - (T / CL))**DL)) * 1000

# ---------- Main Streamlit App ----------

def main():
    st.title("Phi-Phi Method for Liquid-Vapor Mixtures")
    st.write("This app calculates phase equilibrium and then solves a 5×5 system using the phi-phi method.")

    # Sidebar: Constant selection (R)
    st.sidebar.header("Constants")
    R = st.sidebar.selectbox("Select Gas Constant R (m³·Pa/mol·K)", options=[8.31442])
    
    # ---------- Phase Equilibrium Input ----------
    st.header("Phase Equilibrium Inputs")
    T = st.number_input("Temperature (K)", value=291.15, format="%.5f")
    PCO2 = st.number_input("Partial Pressure of CO₂ (Pa)", value=90000)

    # DIPPR parameters (constants used for vapor pressure and liquid density correlations)
    # For CO₂:
    AV = 47.0169
    BV = -2839
    CV = -3.86388
    DV = 2.81115e-16
    EV = 6
    # For CS₂ (vapor pressure correlation):
    Av = 32.308
    Bv = -3813.2
    Cv = -1.5356
    Dv = 3.436e-18
    Ev = 6
    # Liquid density parameters for CO₂:
    A = 2.768
    B = 0.26212
    C = 304.21
    D = 0.2908
    # Liquid density parameters for CS₂:
    AL = 1.561243
    BL = 0.270095
    CL = 552.49
    DL = 0.28571

    # ---------- Initial Calculations for Guess Values ----------
    P_vap_CO2 = Liq_Vapor_PressureCO2(AV, BV, CV, DV, EV, T)
    P_vap_CS2 = Liq_Vapor_PressureCS2(Av, Bv, Cv, Dv, Ev, T)
    X_CO2 = PCO2 / P_vap_CO2
    X_CS2 = 1 - X_CO2
    PCS2 = X_CS2 * P_vap_CS2
    PT = PCO2 + PCS2
    V_liquid = 1 / (X_CO2 * L_rhoCO2(A, B, C, D, T) + X_CS2 * L_rhoCS2(AL, BL, CL, DL, T))
    V_vapor = (R * T) / PT

    st.subheader("Initial Guess Values")
    st.write(f"Liquid-phase mole fraction of CO₂ (X_CO2): {X_CO2:.5f}")
    st.write(f"Liquid-phase mole fraction of CS₂ (X_CS2): {X_CS2:.5f}")
    st.write(f"Total Pressure (PT): {PT:.2f} Pa")
    st.write(f"Liquid molar volume (V_liquid): {V_liquid:.5e} m³/mol")
    st.write(f"Vapor molar volume (V_vapor): {V_vapor:.5e} m³/mol")
    
    # ---------- Critical Properties Inputs for Equation System ----------
    st.header("Critical Properties Inputs")
    Tc_CO2 = st.number_input("Critical Temperature of CO₂ (K)", value=304.21, format="%.5f")
    Pc_CO2 = st.number_input("Critical Pressure of CO₂ (Pa)", value=7383000, format="%.5f")
    Tc_CS2 = st.number_input("Critical Temperature of CS₂ (K)", value=552.49, format="%.5f")
    Pc_CS2 = st.number_input("Critical Pressure of CS₂ (Pa)", value=7329000, format="%.5f")
    
    # Run the full calculation when button is pressed
    if st.button("Run Calculation"):
        # Use initial guess values for the unknowns:
        # unknowns: [x1 (liq mole fraction CO₂), y1 (vap mole fraction CO₂), V_vapor, V_liquid, P_total]
        initial_guess = [X_CO2, X_CS2, V_vapor, V_liquid, PT]

        # For the equation system, define additional variables:
        Tc1 = Tc_CO2
        Pc1 = Pc_CO2
        Tc2 = Tc_CS2
        Pc2 = Pc_CS2
        P11 = PCO2  # Given value for component 1 (CO₂)

        # Calculate b1 and b2
        b1 = 0.07780 * (R * Tc1 / Pc1)
        b2 = 0.07780 * (R * Tc2 / Pc2)

        # ---------- Functions for the Equation System ----------
        def bmixV(y1, y2):
            return y1 * b1 + y2 * b2

        def bmixL(x1, x2):
            return x1 * b1 + x2 * b2

        def zV_mix(V_V, P):
            return (P * V_V) / (R * T)

        def zL_mix(V_L, P):
            return (P * V_L) / (R * T)

        # Parameters for the mixing rules
        w1 = 0.223621
        w2 = 0.084158
        k1 = 0.37464 + 1.54226 * w1 - 0.26992 * (w1**2)
        k2 = 0.37464 + 1.54226 * w2 - 0.26992 * (w2**2)
        alpha1 = (1 + k1 * (1 - np.sqrt(T / Tc1)))**2
        alpha2 = (1 + k2 * (1 - np.sqrt(T / Tc2)))**2
        a11 = 0.45724 * ((R**2) * (Tc1**2) * alpha1 / Pc1)
        a22 = 0.45724 * ((R**2) * (Tc2**2) * alpha2 / Pc2)
        k12 = 0  # Given value
        a12 = np.sqrt(a11 * a22) * (1 - k12)

        def amixV(y1, y2):
            return (y1**2 * a11) + (2 * y1 * y2 * a12) + (y2**2 * a22)

        def amixL(x1, x2):
            return (x1**2 * a11) + (2 * x1 * x2 * a12) + (x2**2 * a22)

        def CO2SUMY(y1, y2):
            return 2 * (y1 * a11 + y2 * a12)

        def CO2SUMX(x1, x2):
            return 2 * (x1 * a11 + x2 * a12)

        def CS2SUMY(y1, y2):
            return 2 * (y2 * a22 + y1 * a12)

        def CS2SUMX(x1, x2):
            return 2 * (x2 * a22 + x1 * a12)

        def phi1_V(V_V, y1, y2, P):
            a_term = b1 / bmixV(y1, y2) * (zV_mix(V_V, P) - 1)
            b_term = np.log(zV_mix(V_V, P) - (bmixV(y1, y2) * P) / (R * T))
            c_term = amixV(y1, y2) / (2 * np.sqrt(2) * bmixV(y1, y2) * R * T)
            d_term = CO2SUMY(y1, y2) / amixV(y1, y2) - b1 / bmixV(y1, y2)
            e_term = np.log(
                (zV_mix(V_V, P) + (1 + np.sqrt(2)) * (bmixV(y1, y2) * P / (R * T))) /
                (zV_mix(V_V, P) + (1 - np.sqrt(2)) * (bmixV(y1, y2) * P / (R * T)))
            )
            ln_phi = a_term - b_term - c_term * d_term * e_term
            return np.exp(ln_phi)

        def phi1_L(V_L, x1, x2, P):
            a_term = b1 / bmixL(x1, x2) * (zL_mix(V_L, P) - 1)
            b_term = np.log(zL_mix(V_L, P) - (bmixL(x1, x2) * P) / (R * T))
            c_term = amixL(x1, x2) / (2 * np.sqrt(2) * bmixL(x1, x2) * R * T)
            d_term = CO2SUMX(x1, x2) / amixL(x1, x2) - b1 / bmixL(x1, x2)
            e_term = np.log(
                (zL_mix(V_L, P) + (1 + np.sqrt(2)) * (bmixL(x1, x2) * P / (R * T))) /
                (zL_mix(V_L, P) + (1 - np.sqrt(2)) * (bmixL(x1, x2) * P / (R * T)))
            )
            ln_phi = a_term - b_term - c_term * d_term * e_term
            return np.exp(ln_phi)

        def phi2_V(V_V, y1, y2, P):
            a_term = b2 / bmixV(y1, y2) * (zV_mix(V_V, P) - 1)
            b_term = np.log(zV_mix(V_V, P) - (bmixV(y1, y2) * P) / (R * T))
            c_term = amixV(y1, y2) / (2 * np.sqrt(2) * bmixV(y1, y2) * R * T)
            d_term = CS2SUMY(y1, y2) / amixV(y1, y2) - b2 / bmixV(y1, y2)
            e_term = np.log(
                (zV_mix(V_V, P) + (1 + np.sqrt(2)) * (bmixV(y1, y2) * P / (R * T))) /
                (zV_mix(V_V, P) + (1 - np.sqrt(2)) * (bmixV(y1, y2) * P / (R * T)))
            )
            ln_phi = a_term - b_term - c_term * d_term * e_term
            return np.exp(ln_phi)

        def phi2_L(V_L, x1, x2, P):
            a_term = b2 / bmixL(x1, x2) * (zL_mix(V_L, P) - 1)
            b_term = np.log(zL_mix(V_L, P) - (bmixL(x1, x2) * P) / (R * T))
            c_term = amixL(x1, x2) / (2 * np.sqrt(2) * bmixL(x1, x2) * R * T)
            d_term = CS2SUMX(x1, x2) / amixL(x1, x2) - b2 / bmixL(x1, x2)
            e_term = np.log(
                (zL_mix(V_L, P) + (1 + np.sqrt(2)) * (bmixL(x1, x2) * P / (R * T))) /
                (zL_mix(V_L, P) + (1 - np.sqrt(2)) * (bmixL(x1, x2) * P / (R * T)))
            )
            ln_phi = a_term - b_term - c_term * d_term * e_term
            return np.exp(ln_phi)

        def equations(vars):
            # Unknowns: x1, y1, V_vapor, V_liquid, P_total
            x1, y1, V_V, V_L, P = vars
            x2 = 1 - x1
            y2 = 1 - y1
            P1 = PCO2  # Given: P11 = PCO2
            eq1 = ((R * T) / (V_V - bmixV(y1, y2)) - 
                   amixV(y1, y2) / ((V_V * (V_V + bmixV(y1, y2))) + (bmixV(y1, y2) * (V_V - bmixV(y1, y2))))) - P
            eq2 = ((R * T) / (V_L - bmixL(x1, x2)) - 
                   amixL(x1, x2) / ((V_L * (V_L + bmixL(x1, x2))) + (bmixL(x1, x2) * (V_L - bmixL(x1, x2))))) - P
            eq3 = (x1 * phi1_L(V_L, x1, x2, P)) - (y1 * phi1_V(V_V, y1, y2, P))
            eq4 = (x2 * phi2_L(V_L, x1, x2, P)) - (y2 * phi2_V(V_V, y1, y2, P))
            eq5 = (y1 * P) - P1
            return [eq3, eq4, eq1, eq2, eq5]

        # Solve the 5x5 system using the initial guesses
        solution, infodict, ier, mesg = fsolve(equations, initial_guess, full_output=True)
        if ier == 1:
            x1, y1, V_V, V_L, P_sol = solution
            x2 = 1 - x1
            y2 = 1 - y1
            st.subheader("Equation System Solution")
            st.write(f"Liquid-phase mole fraction of CO₂ (x1): {x1:.5f}")
            st.write(f"Liquid-phase mole fraction of CS₂ (x2): {x2:.5f}")
            st.write(f"Vapor-phase mole fraction of CO₂ (y1): {y1:.5f}")
            st.write(f"Vapor-phase mole fraction of CS₂ (y2): {y2:.5f}")
            st.write(f"Vapor molar volume (V_vapor): {V_V:.5e} m³/mol")
            st.write(f"Liquid molar volume (V_liquid): {V_L:.5e} m³/mol")
            st.write(f"Total Pressure (P): {P_sol:.2f} Pa")
        else:
            st.error("Solution did not converge. Please check your inputs or try different initial guesses.")

if __name__ == "__main__":
    main()
