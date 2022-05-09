
% Computes the step length
%{
INPUT
    - w: Vector of optimal weights that minimize the Loss Function.
    - Xte: Matrix of test samples, that will only be used to compute 'te_acc'
       once the model is fitted.
    - yte: Binary output vector representing whether the i-th test sample is
       contained in 'num_target'.
    - la: Regulatization parameter of the Ridge Regression.
    - L: objective function
    - gL: gradient of the objective function
    - hL: hessian of the objective function
    - d: descendent direction
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
    - al: step length
    - IWout: indicador de quines condicions de Wolf compleix
            iWout = 0: al does not satisfy any WC
            iWout = 1: al satisfies (WC1)
            iWout = 2: al satisfies WC
            iWout = 3: al satisfies SWC
    - iout: indicator of the stoppping cause of BLS
            iout = 0: succeed
            iout = 1: kmax exceeded
            iout = 2: epsal exceeded
%}
function [al, iWout, iout] = STEP_L(w, Xtr, ytr, la, L, gL, hL, d, almax, ialmin, rho, c1, c2, ils, kBLSmax, epsal)
    iWout = []; 
    if ils == 0 % ELS 
        al = -(gL(w)'*d)/(d'*hL(w)*d); 
        iWout = 5; 

    elseif ils == 1 % BLS
        al = almax; 
        [satisfy, iWout] = WOLFE(w, al, d, L, gL, c1, c2, ils);
        while ~satisfy & al >= ialmin
            al = rho*al; 
            [satisfy, iWout] = WOLFE(w, al, d, L, gL, c1, c2, ils);
        end

    elseif ils == 3 % BLSNW32
        [al,iout] = uo_BLSNW32(@(w) L(w, Xtr, ytr, la), @(w) gL(w, Xtr, ytr, la), w, d, almax, c1, c2, kBLSmax, epsal);
    end
end