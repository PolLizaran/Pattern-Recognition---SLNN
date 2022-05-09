% Computes the optimal point using the BFGS method --> dk = -H(x)g(x)
%{
INPUT
    - w_0: Vector of optimal weights that minimize the Loss Function.
    - Xte: Matrix of test samples, that will only be used to compute 'te_acc'
       once the model is fitted.
    - yte: Binary output vector representing whether the i-th test sample is
       contained in 'num_target'.
    - la: Regulatization parameter of the Ridge Regression.
    - L: objective function
    - gL: gradient of the objective function
    - hL: hessian of the objective function
    - epsG: Minimum value of the gradient norm as a stopping criteria.
    - kmax: Maximum number of iterations as a stopping criteria.
    - ialmax: Maximum step length.
    - ialmin: Minimum step length.
    - rho: Constant for BLS.
    - c1,c2: Wolfe constants. 
    - ils: Parameter that indicates the line search method to be used. 
       (ils = 1 -> ELS, ils = 2 -> BLS, ils = 3 -> BLSNW32)    
    - kmaxBLS: Maximum number of iterations as a stopping criteria of BLSNW32.
    - epsal: Minimum variation between consecutive reductions of k-th step length.
OUTPUT
    - wk: vector of points (= vector of parameters)
    - k: number of iterations 
%}
function [wk] = BFGS(w_0, Xtr, ytr, la, L, gL, hL, epsG, kmax, ialmax, ...
                                              ialmin, rho, c1, c2, ils, kBLSmax, epsal)

    n = length(w_0); I = eye(n); H = I; % Inicialització de la matriu H
    wk = [w_0]; dk = []; alk = []; iWk =[]; betak = []; Hk = [H]; k = 0; 
    ioutk = []; w = w_0;

    while norm(gL(w, Xtr, ytr, la)) > epsG & k < kmax
        % Descendent direction
        d = -H * gL(w, Xtr, ytr, la); dk = [dk, d];
        
        % Computation of the step length
        if k == 0
            almax = 1;
        else 
            if ialmax == 1
                almax = al*(gL(wk(:,end - 1), Xtr, ytr, la)'*dk(:,end-1))/(gL(w, Xtr, ytr, la)'*d);
            elseif ialmax == 2
                almax = 2*(L(w, Xtr, ytr, la)-L(wk(:,end-1), Xtr, ytr, la))/(gL(w, Xtr, ytr, la)'*d);
            end
        end
        [al, iWout, iout] = STEP_L(w, Xtr, ytr, la, L, gL, hL, d, almax, ialmin, rho, c1, c2, ils, kBLSmax, epsal);
        
       % Update of the vector of parameters
        w = w + d*al; 
        wk = [wk, w]; alk = [alk, al]; iWk = [iWk, iWout];

        s = wk(:, end) - wk(:, end - 1); % Secant equation
        Dif_grad = gL(wk(:,end), Xtr, ytr, la) - gL(wk(:,end - 1), Xtr, ytr, la); % Gradient substraction
        p = 1/(Dif_grad'*s); 
        H = (I - p*s*Dif_grad')*H*(I - p*Dif_grad*s') + p*s*s'; Hk(:,:,end+1) = [H]; % Hessian aproximation
        k = k + 1; 
    end
end
