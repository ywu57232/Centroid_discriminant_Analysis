import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from math import sqrt, exp, log
from scipy.stats import norm


def BOImpl(objectiveFunc, XRange, maxIter, seed):
    np.random.seed(seed)

    LB, UB = XRange[0], XRange[1]

    # ========== 1) Initial random samples ========== #
    nInitialSamples = 4
    XBest = LB + np.random.rand(nInitialSamples) * (UB - LB)

    bsf = min_pairwise_dist(XBest.reshape(-1, 1))  # best-so-far
    reps = 50
    for r in range(1, reps + 1):
        np.random.seed(seed * 100 + r)
        Xtemp = LB + np.random.rand(nInitialSamples) * (UB - LB)
        dist_temp = min_pairwise_dist(Xtemp.reshape(-1, 1))
        if dist_temp > bsf:
            bsf = dist_temp
            XBest = Xtemp

    # Evaluate objective at these initial points
    xSamples = XBest.copy()
    ySamples = np.array([objectiveFunc(x) for x in xSamples])

    numLoop = maxIter - nInitialSamples + 1
    if numLoop < 1:
        numLoop = 1

    for iteration in range(numLoop):
        SigmaLowerBound = max(1e-8, np.std(ySamples) * 1e-2)
        sigma0 = max(SigmaLowerBound, np.std(ySamples) / 5)
        theta0 = np.array([log(1.0), log(90.0), log(sigma0 - SigmaLowerBound)], dtype=float)

        def evaluateTheta(theta):
            f, grad = NegLikelihood_and_gradient(xSamples, ySamples, theta, SigmaLowerBound)
            return (f, grad)

        opts = {
            "maxiter": 50,
            "disp": False,
            "gtol": 1e-3
        }
        result = minimize(
            fun=evaluateTheta,
            x0=theta0,
            method="BFGS",
            jac=True,
            options=opts
        )
        thetaHat = result.x

        n = len(xSamples)
        K = Matern52Kernel(xSamples, xSamples, thetaHat, gradient_mode=False)

        noise_var = (SigmaLowerBound + np.exp(thetaHat[2])) ** 2
        K = K + noise_var * np.eye(n)

        np.random.seed(seed * 50 + iteration)
        xGridBase = np.linspace(LB, UB, 360)
        xGrid = xGridBase + 0.25 * (np.random.rand(360) - 0.5) * 2

        kGrid = Matern52Kernel(xSamples, xGrid, thetaHat, gradient_mode=False)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, ySamples))

        mu = kGrid.T @ alpha
        sigmaF2 = np.exp(2 * thetaHat[0])
        solveKG = np.linalg.solve(L, kGrid)
        solveKG = np.linalg.solve(L.T, solveKG)
        diag_part = np.einsum("ij,ij->j", solveKG, kGrid)
        sigma_vals = sigmaF2 - diag_part
        sigma_vals = np.maximum(0, sigma_vals)
        sigma_vals = np.sqrt(sigma_vals)

        bestY = np.min(ySamples)

        Z = (bestY - mu) / sigma_vals
        ei = sigma_vals * (Z * norm.cdf(Z) + norm.pdf(Z))
        ei[sigma_vals == 0] = 0

        maxIdx = np.argmax(ei)
        nextX = xGrid[maxIdx]

        nextY = objectiveFunc(nextX)

        xSamples = np.append(xSamples, nextX)
        ySamples = np.append(ySamples, nextY)

    bestMuIdx = np.argmin(mu)
    Mdl = {}
    Mdl["YAtMinEstimatedObjective"] = float(mu[bestMuIdx])
    Mdl["XAtMinEstimatedObjective"] = float(xGrid[bestMuIdx])
    return Mdl



def NegLikelihood_and_gradient(X, y, theta, sigmaLowerBound):
    sigma = sigmaLowerBound + np.exp(theta[2])

    K, dK = Matern52KernelWithDeriv(X, X, theta)
    n = len(X)
    K[np.diag_indices(n)] += sigma ** 2

    L = np.linalg.cholesky(K)

    Linvy = np.linalg.solve(L, y)
    ones_n = np.ones(n)
    LinvH = np.linalg.solve(L, ones_n)
    numerator = LinvH @ Linvy
    denominator = LinvH @ LinvH
    betaHat = numerator / denominator

    LinvyRemoveBeta = Linvy - LinvH * betaHat

    first_term = -0.5 * np.dot(LinvyRemoveBeta, LinvyRemoveBeta)
    second_term = -(n / 2) * np.log(2 * np.pi)
    third_term = -np.sum(np.log(np.diag(L)))
    negLogLikelihood = -(first_term + second_term + third_term)

    alphaHat = np.linalg.solve(L.T, LinvyRemoveBeta)

    Linv = np.linalg.inv(L)

    gradNegLogLikelihood = np.zeros(3)

    dKsigmaF = dK[0]
    dKlength = dK[1]

    quad_sf = 0.5 * np.dot(alphaHat, (dKsigmaF @ alphaHat))
    tmp_sf = np.linalg.solve(L, dKsigmaF)
    detKTerm_sf = -0.5 * np.sum((Linv * tmp_sf.T))

    grad_sf = -(quad_sf + detKTerm_sf)

    dKls = dKlength
    quad_ls = 0.5 * np.dot(alphaHat, (dKls @ alphaHat))
    tmp_ls = np.linalg.solve(L, dKls)
    detKTerm_ls = -0.5 * np.sum((Linv * tmp_ls.T))

    grad_ls = -(quad_ls + detKTerm_ls)

    sigmaT = sigma * (sigma - sigmaLowerBound)

    quad_sn = sigmaT * np.dot(alphaHat, alphaHat)
    detKTerm_sn = -sigmaT * np.sum(Linv * Linv)
    grad_sn = -(quad_sn + detKTerm_sn)

    gradNegLogLikelihood[0] = grad_sf
    gradNegLogLikelihood[1] = grad_ls
    gradNegLogLikelihood[2] = grad_sn

    return negLogLikelihood, gradNegLogLikelihood


def Matern52Kernel(X1, X2, theta, gradient_mode=True):
    X1 = np.array(X1, dtype=float).reshape(-1, 1)
    X2 = np.array(X2, dtype=float).reshape(-1, 1)

    sigmaF = max(exp(theta[0]), 1e-6)
    sigmaL = max(exp(theta[1]), 1e-6)

    dSquare = cdist_sqeuclidean(X1, X2)
    r = sqrt(5) * np.sqrt(dSquare) / sigmaL
    K = sigmaF ** 2 * (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)

    if gradient_mode:
        return K
    else:
        return K


def Matern52KernelWithDeriv(X1, X2, theta):
    X1 = np.array(X1, dtype=float).reshape(-1, 1)
    X2 = np.array(X2, dtype=float).reshape(-1, 1)

    sigmaF = max(exp(theta[0]), 1e-6)
    sigmaL = max(exp(theta[1]), 1e-6)

    dSquare = cdist_sqeuclidean(X1, X2)

    r = sqrt(5) * np.sqrt(dSquare) / sigmaL
    K = sigmaF ** 2 * (1.0 + r + (r ** 2) / 3.0) * np.exp(-r)

    dKsigmaF = 2.0 * K

    with np.errstate(divide='ignore', invalid='ignore'):
        # avoid warnings if r=0
        f_r = (1 + r + r ** 2 / 3.0) * np.exp(-r)
        dKsigmaL = (5.0 / 3.0) * (dSquare / (sigmaL ** 2)) * f_r

    return K, [dKsigmaF, dKsigmaL]


def setOpts():
    return {
        "Display": 'off',
        "MaxFunEvals": None,
        "MaxIter": 50,
        "TolBnd": None,
        "TolFun": 1e-3,
        "TolTypeFun": None,
        "TolX": 1e-3,
        "TolTypeX": None,
        "GradObj": 'on'
    }

def cdist_sqeuclidean(X1, X2):
    diff = X1 - X2.T
    return diff ** 2


def min_pairwise_dist(X):
    if X.shape[0] < 2:
        return 0.0
    d = pdist(X, metric='euclidean')
    return np.min(d)
