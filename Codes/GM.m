function [wk] = GM(w_0, Xtr, ytr, la, L, gL, hL, epsG, kmax, ialmax, ialmin, ...
                                        rho, c1, c2, ils, kBLSmax, epsal)
    %{ 
    Function using Gradient Method (takes d = -g(x)). Here the gradient of the function to minimize, 
    Loss function of the outcome of the sigmoids of the Neuronal Network, is denoted by 'gL'. The step
    descent direction is computed preferably with BLSNW32.
    %} 
    wk = [w_0]; w = w_0;
    dk = []; alk = []; iWk =[]; k = 0; ioutk = [];
    while norm(gL(w, Xtr, ytr, la)) >= epsG & k < kmax
        d = -gL(w, Xtr, ytr, la); dk = [dk, d]; 

        if k == 0                                     % Computation of almax depending on ialmax value
            almax = 1;
        else 
            if ialmax == 1
                almax = al*(gL(wk(:,end - 1), Xtr, ytr, la)'*dk(:,end-1))/(gL(w, Xtr, ytr, la)'*d);
            elseif ialmax == 2
                 almax = 2*(L(w, Xtr, ytr, la)-L(wk(:,end-1),Xtr, ytr, la))/(gL(w, Xtr, ytr, la)'*d);
            end
        end

        [al, iWout, iout] = STEP_L(w, Xtr, ytr, la, L, gL, hL, d, almax, ialmin, rho, c1, c2, ils, ...
                                   kBLSmax, epsal);
        w = w + d*al;                                 % w^{k+1}
        wk = [wk, w];  alk = [alk, al]; iWk = [iWk, iWout]; ioutk = [ioutk, iout];
        k = k + 1;
    end
end