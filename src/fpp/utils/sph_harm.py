def Ylm(l, m, theta, phi): 
    """Redefine spherical harmonics because scipy.special cannot keep it together."""

    if abs(m) > l:
        raise ValueError("Absolute value of m must be less than or equal to l.")

    try:
        from scipy.special import sph_harm_y
        return sph_harm_y(l, m, theta, phi)
    except ImportError:
        from scipy.special import sph_harm
        return sph_harm(m, l, theta, phi)