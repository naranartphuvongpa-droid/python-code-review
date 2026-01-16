import math

def chamis_function(x):
    Kf = 0.5          # fiber volume fraction
    Km = 1 - Kf       # matrix fraction
    Ef11 = 420        # GPa
    Em = 425          # GPa
    Vf12 = 0.18
    Vm = 0.18
    af11 = 0.1e-6
    am = 4.25e-6
    Gm = 170          # GPa
    Gf12 = 126        # GPa
    

    # Longitudinal modulus
    E1 = Kf*Ef11 + Km*Em

    # In-plane modulus
    Ef22 = 84
    El22 = Em/(1 - (math.sqrt(Kf)*(1 - Em/Ef22)))
    E2 = (1 - math.sqrt(Kf))*Em + math.sqrt(Kf)*El22

    # V12 stays same
    V12 = Kf*Vf12 + Km*Vm

    # Corrected V23 (limit between 0–1)
    Vf23 = 0.15
    Vl12 = 0.1
    V23_calc = Kf*Vf23 + Km*(2*Vm - Vl12*El22/E1)
    V23 = max(0.0, min(V23_calc, 1.0))  # clamp to 0–1

    # Shear moduli
    Gl12 = Gm / (1 - (math.sqrt(Kf)*(1 - Gm/Gf12)))
    G12 = (1 - math.sqrt(Kf))*Gm + math.sqrt(Kf)*Gl12
    G23 = Gm / (1 - (math.sqrt(Kf)*(1 - Gm/Gf12)))

    # Thermal expansion
    al11 = (Kf*af11*Ef11 + Km*am*Em)/E1
    al12 = (math.sqrt(Kf) + (1 - math.sqrt(Kf))*(1 + Kf*Vm*Ef11/E1))*am

    E3=E2
    V13=V23
    G13=G23

    return E1, V12, E2, V23, G23, G12, E3, V13, G13, al11, al12



