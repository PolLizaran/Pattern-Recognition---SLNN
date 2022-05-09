function [satisfy, iWout] = WOLFE(w, al, d, L, gL, c1, c2, ils)
    iWout = 0;
    satisfy = false;
    if L(w + al*d) <= L(w) + c1*gL(w)'*d*al %satisfies WC1
        iWout = 1;
        if ils == 1 
            if gL(w + al*d)'*d >= c2*gL(w)'*d %satisfies WC2
                iWout = 2;
                satisfy = true;
            end
        elseif ils == 2 
            if abs(gL(w + al*d)'*d) <= c2*abs(gL(w)'*d)%satisfies SWC2
                iWout = 3;
                satisfy = true;
            end
        end
    end
end  