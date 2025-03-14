cd ~/Git/blopex

addpath("jacobi/")
addpath("blopex_tools/matlab/lobpcg/")
addpath("~/Dropbox/Git/matrix-market/")

[A, rows, cols, entries] = mmread("Poisson_SExp_sig21.0_L0.1_DoF4000_K.mtx");
%[A, rows, cols, entries] = mmread("Poisson_SExp_sig21.0_L0.1_DoF8000_K.mtx");
%[A, rows, cols, entries] = mmread("Poisson_SExp_sig21.0_L0.1_DoF16000_K.mtx");
%[A, rows, cols, entries] = mmread("Poisson_SExp_sig21.0_L0.1_DoF32000_K.mtx");
%[A, rows, cols, entries] = mmread("Poisson_SExp_sig21.0_L0.1_DoF64000_K.mtx");
%[A, rows, cols, entries] = mmread("Poisson_SExp_sig21.0_L0.1_DoF128000_K.mtx");

[n, ~] = size(A);
m = 10;

B = [];
T = spdiags(1 ./ diag(A), 0, size(A,1), size(A,2));
residualTolerance = 1e-2;
maxIterations = 10000;
verbosityLevel = 1;

rng(42);

X = rand(n, m);
[X, lambda, failureFlag, lambdaHistory, residualNormHistory] = lobpcg(X, A, B, T, ...
                                                                      residualTolerance, ...
                                                                      maxIterations, ...
                                                                      verbosityLevel);

[~, it1] = size(residualNormHistory);
it1

[X, lambda, failureFlag, lambdaHistory, residualNormHistory] = lobpcg_jacobi(X, A, B, T, ...
                                                                             residualTolerance, ...
                                                                             maxIterations, ...
                                                                             verbosityLevel);
[~, it2] = size(residualNormHistory);
it2


A = rand(4, 4); A = A + A';
tol = 1e-6;
[V, D, hist] = jacobi(A, 1, tol);