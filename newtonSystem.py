import numpy as np
import matplotlib.pyplot as plt

# Task 1: Define the system f(x) and its Jacobian J(x)

def f(x):
    x1, x2, x3 = x
    return np.array([
        6*x1 - 2*np.cos(x2*x3) - 1,
        9*x2 + np.sqrt(x1**2 + np.sin(x3) + 1.06) + 0.9,
        60*x3 + 3*np.exp(x1*x2) + 10*np.pi - 3
    ])


def J(x):
    x1, x2, x3 = x
    # partial derivatives
    df1_dx1 = 6
    df1_dx2 = 2 * x3 * np.sin(x2*x3)
    df1_dx3 = 2 * x2 * np.sin(x2*x3)

    df2_dx1 = x1 / (np.sqrt(x1**2 + np.sin(x3) + 1.06))
    df2_dx2 = 9
    df2_dx3 = np.cos(x3) / (np.sqrt(x1**2 + np.sin(x3) + 1.06))

    df3_dx1 = 3 * x2 * np.exp(x1*x2)
    df3_dx2 = 3 * x1 * np.exp(x1*x2)
    df3_dx3 = 60

    return np.array([
        [df1_dx1, df1_dx2, df1_dx3],
        [df2_dx1, df2_dx2, df2_dx3],
        [df3_dx1, df3_dx2, df3_dx3]
    ])


def is_finite(vec, name):
    if not np.all(np.isfinite(vec)):
        print(f"Divergencia detectada: {name} contiene valores no finitos.")
        return False
    return True

# Task 2: Newton-Raphson method for systems
def newton_system(residual, jacobian, x0, tol=1e-12, maxiter=50, tol_dx=1e-12):
    x = x0.astype(float)
    history = [x.copy()]
    for k in range(maxiter):
        
        F = residual(x)
        if (not is_finite(F, f"F (iter {k})")):
            break
        
        Jx = jacobian(x)
        if (not is_finite(Jx, f"J (iter {k})")):
            break
        
        
        try:
            dx = np.linalg.solve(Jx, -F)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Jacobian is singular at iteration {k}")
        
        if (not is_finite(dx, f"dx (iter {k})")):
            break
        
        x = x + dx
        history.append(x.copy())
        if np.linalg.norm(F, ord=2) < tol and np.linalg.norm(dx, ord=2) < tol_dx:
            break
    return x, np.array(history)

# Task 5: Modified Newton-Raphson (fixed Jacobian at x0)
def modified_newton(residual, jacobian, x0, tol=1e-12, maxiter=50, tol_dx=1e-12):
    x = x0.astype(float)
    J0 = jacobian(x0)
    # LU factorization could be used here; we simply solve repeatedly
    history = [x.copy()]
    for k in range(maxiter):
        
        F = residual(x)
        if(not is_finite(F, f"F (iter {k})")):
            break
        
        try:
            dx = np.linalg.solve(J0, -F)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Fixed Jacobian is singular")
        
        if (not is_finite(dx, f"dx (iter {k})")):
            break
        
        x = x + dx
        history.append(x.copy())
        if np.linalg.norm(F, ord=2) < tol and np.linalg.norm(dx, ord=2) < tol_dx:
            break
    return x, np.array(history)

# Task 3: Interval analysis for initial guesses
# From eq1: 6*x1 -2*cos(x2*x3) -1 = 0 -> cos in [-1,1]
#   6*x1 -2*[-1,1] -1 = 0 -> 6*x1 - [-2,2] -1 = 0 -> 6*x1 = 1 + [-2,2] -> 6*x1 in [-1,3] -> x1 in [-1/6, 0.5]
x1_interval = (-1/6, 0.5)
# From eq2: 9*x2 + |x1| + sin(x3) + 1.06 + 0.9 = 0, sin in [-1,1]
#   9*x2 + |x1| + [-1,1] + 1.96 = 0 -> 9*x2 = -1.96 - |x1| + [-1,1] -> range depends on x1
#   Using max |x1|=0.5: 9*x2 in [-1.96-0.5-1, -1.96-0+1] = [-3.46, -0.96] -> x2 in [-0.384, -0.107]
x2_interval = (-0.384, -0.107)
# From eq3: 60*x3 + 3*exp(x1*x2) + 10*pi -3 =0, exp in [exp(-0.192), exp(0)] ~ [0.825,1]
#   60*x3 = -3*exp +3 -10*pi -> range approx [ -3*1 +3 -10π, -3*0.825+3 -10π ]
#   = [3-3-10π, 3-2.475-10π] = [-10π, 0.525-10π] ≈ [-31.416, -30.891] -> x3 ≈ [-0.524, -0.515]
x3_interval = (-np.pi/6, -0.515)

# Choose a reasonable initial guess near center of intervals:
x0 = np.array([0.2, -0.25, -0.52])

# Task 4 & 6: Solve for different initial guesses and plot convergence
initials = {
    'x0_zero': np.array([0.,0.,0.]),
    'x0_one': np.array([1.,1.,1.]),
    'x0_five': np.array([5.,5.,5.]),
    'x0_mix': np.array([-15.,15.,-15.]),
    'x0_interval': x0
}
results = {}
for name, x_init in initials.items():
    sol_n, hist_n = newton_system(f, J, x_init)
    sol_m, hist_m = modified_newton(f, J, x_init)
    # Errores en cada iteración
    errs_n = np.linalg.norm([f(x) for x in hist_n], axis=1)
    errs_m = np.linalg.norm([f(x) for x in hist_m], axis=1)

    # convergencia cuadratica
    its = np.arange(max(len(errs_n), len(errs_m)))
     


    # Plot individual
    plt.figure()
    plt.loglog(errs_n, marker='o', label='Newton completo')
    plt.loglog(errs_m, marker='x', label='Newton modificado')
    plt.xlabel('Iteración')
    plt.ylabel('||f(x)||')
    plt.title(f'Convergencia para {name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"{name}: Newton completo -> {len(hist_n)-1} iter., Newton modificado -> {len(hist_m)-1} iter.")