function [w_opt, wk, k] = SGM(w_0, la, L, gL, Xtr, ytr, Xte, yte, sg_al0, sg_be, sg_ga, sg_emax, ...
                              sg_ebest, sg_seed)
    %{
    This function runs the SGM algorithm. It actually estimates the gradient
    of the Loss function by taking a subset of the training data subset. 
    %}
    if ~isempty(sg_seed), rng(sg_seed); end                 % Set seed for reproducibility
    p = size(Xtr, 2);                                       % Training data set size
    m = floor(sg_ga * p);                                   % Minibatch size
    sg_ke = ceil(p/m); sg_kmax = sg_emax * sg_ke;           % Maximum number of iterations to stop
    sg_al = 0.01 * sg_al0; sg_k = floor(sg_be * sg_kmax);   % Parameters to compute step length;
    best_Loss = inf;                                        % Loss not defined at begining
    w_opt = w_0; wk = [w_0]; w = w_0;                       % Best solution so far at the begining
    e = 0;  s = 0; k = 0;                                   % Indexes of loops
    while(e <= sg_emax && s < sg_ebest && k < sg_kmax)
        perm_Xtr = randperm(p);                                         % Indexes of permuted columns
        for i = 0:ceil((p/m) - 1) 
            Minibatch = perm_Xtr(m*i + 1 : min(m*i + m, p));            % Column subset of Xtr 
            Xtr_mini = Xtr(:,Minibatch); ytr_mini = ytr(Minibatch);                                              
            
            d = -gL(w, Xtr_mini, ytr_mini, la);                         % Descent direction
            
            if k <= sg_k                                                % Step length computation
                al = (1 - (k/sg_k))*sg_al0 + (k/sg_k)*sg_al;
            else
                al = sg_al;
            end
            w = w + al*d; wk = [wk, w]; k = k + 1;                      % w^{k+1}
        end
        e = e + 1;                                                      % Next epoch
        Lte_value = L(w, Xte, yte, la); 
        if Lte_value < best_Loss
            best_Loss = Lte_value;
            w_opt = w;                                                  % Optimal vector solution so far
            s = 0;
        else
            s = s + 1;
        end
    end
end 