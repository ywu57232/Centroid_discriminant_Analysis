function Mdl = BOImpl(objectiveFunc, XRange, maxIter, seed)
%% A Bayesian Optimization with Reduced Overhead

% Define the range of the input
LB = XRange(1);
UB = XRange(2);

% Initial random samples
nInitialSamples = 4;
rng(seed)
XBest = LB + rand(nInitialSamples, 1) * (UB - LB);
bsf = min(pdist(XBest));
reps = 50;
for r = 1:reps
    rng(seed * 100 + r)
    X = LB + rand(nInitialSamples, 1) * (UB - LB);
    if min(pdist(X)) > bsf
        bsf = min(pdist(X));
        XBest = X;
    end
end
xSamples = XBest;
ySamples = arrayfun(objectiveFunc, xSamples);

% Bayesian Optimization Loop
for iter = 1:maxIter -nInitialSamples + 1  
    SigmaLowerBound = max(1e-8, std(ySamples)*1e-2);
    sigma0 = max(SigmaLowerBound, std(ySamples) / 5);
    theta0 = [log(1), log(90), log(sigma0 - SigmaLowerBound)];

    evaluateTheta = @(theta) NegLikelihood_and_gradient(xSamples, ySamples, theta, SigmaLowerBound);

    % Quasi-Newton method with SR1 Hessian update and with trust region
    % strategy for gradients
    [thetaHat, ~] = classreg.learning.gputils.fminqn(evaluateTheta, theta0, 'Options', setOpts(), 'InitialStepSize', []);

    % Construct the covariance matrix K
    n = length(xSamples);
    K = Matern52Kernel(xSamples, xSamples, thetaHat, 'nogradient');
    K = K + (SigmaLowerBound+exp(thetaHat(3)))^2 * eye(n); % Add noise variance for numerical stability

    % Predict the mean and variance over the grid
    rng(seed*50+iter)
    xGrid = linspace(LB, UB, 360)' + 0.25 * (rand(360, 1)-0.5)*2;
    kGrid = Matern52Kernel(xSamples, xGrid, thetaHat, 'nogradient');
    
    mu = kGrid' * (K \ ySamples);
    sigma = sqrt(max(0, exp(2*thetaHat(1)) - diag(kGrid' * (K \ kGrid))));

    % Acquisition function: Expected Improvement
    bestY = min(ySamples);
    Z = (bestY - mu) ./ sigma;
    ei = sigma .* (Z .* normcdf(Z) + normpdf(Z));
    ei(sigma == 0) = 0; % Handle numerical issues

    % Select the next point to evaluate
    [~, maxIdx] = max(ei);
    nextX = xGrid(maxIdx);

    % Evaluate the objective function at the new point
    nextY = objectiveFunc(nextX);

    % Update samples
    xSamples = [xSamples; nextX];
    ySamples = [ySamples; nextY];
end
[Mdl.YAtMinEstimatedObjective, idx_YAtMinEstimatedObjective] = min(mu);
Mdl.XAtMinEstimatedObjective = xGrid(idx_YAtMinEstimatedObjective);
end


function [negLogLikelihood, gradNegLogLikelihood] = NegLikelihood_and_gradient(X, y, theta, sigmaLowerBound)
sigma = sigmaLowerBound + exp(theta(3));

[K,dK] = Matern52Kernel(X, X, theta, []);
N = size(X, 1);
K(1:N+1:N^2) = K(1:N+1:N^2) + sigma^2;
L = chol(K,'lower');

Linvy   = L \ y;
LinvH   = L \ ones(N,1);
betaHat = LinvH \ Linvy;
LinvyRemoveBeta = (Linvy - LinvH * betaHat);
negLogLikelihood = (-0.5 * (LinvyRemoveBeta' * LinvyRemoveBeta) - (N/2)*log(2*pi) - sum(log(abs(diag(L))))) * -1;

% gradient of negative log likelihood
alphaHat = L' \ LinvyRemoveBeta;
Linv = inv(L);
gradNegLogLikelihood = zeros(3,1);
for m = 1:2
    dKm = dK{m};
    quadratic   = 0.5 * (alphaHat' * dKm*alphaHat);
    dKm        = L \ dKm;
    detKTerm  = -0.5 * (Linv(:)' * dKm(:));
    gradNegLogLikelihood(m) = (quadratic + detKTerm) * -1;
end
sigmaT = sigma * (sigma - sigmaLowerBound);
quadratic = sigmaT * (alphaHat' * alphaHat);
detKTerm = -sigmaT * (Linv(:)' * Linv(:));
gradNegLogLikelihood(3) = (quadratic + detKTerm) * -1;
end


function [K, dK] = Matern52Kernel(X1, X2, theta, mode)
dSquare = pdist2(X1, X2,"squaredeuclidean");

sigmaF = max(exp(theta(1)), 1e-6);
sigmaL = max(exp(theta(2)), 1e-6);
K = dSquare / sigmaL.^2;

T   = sqrt(5)*sqrt(K);
K = (sigmaF^2)*((1 + T.*(1 + T/3)).*exp(-T));
if strcmp(mode, 'nogradient')
    return
end

% get grdient
T = ((1 + T).*K)./((1 + T + T.^2/3));
dK{1} = 2*K; % dK / dSigmaF * SigmaF
dK{2} = (5/3)*(dSquare / sigmaL^2 .* T); % dK / dSigmaL * SigmaL
end


function opts = setOpts()
opts = struct(   ...
    "Display", "off", ...
    "MaxFunEvals", [], ...
    "MaxIter", 50, ...
    "TolBnd", [], ...
    "TolFun", 1e-3, ...
    "TolTypeFun", [], ...
    "TolX", 1e-3, ...
    "TolTypeX", [], ...
    "GradObj", 'on', ...
    "Jacobian", [], ...
    "DerivStep", [], ...
    "FunValCheck", [], ...
    "Robust", [], ...
    "RobustWgtFun", [], ...
    "WgtFun", [], ...
    "Tune", [], ...
    "UseParallel", [], ...
    "UseSubstreams", [], ...
    "Streams", [], ...
    "OutputFcn", [] ...
    );
end