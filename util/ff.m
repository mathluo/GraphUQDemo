function ans=ff(u,yy,p,q)

if sign(u)>0
    if sign(yy)>0
     ans=p;
    else
        ans=1-p;
    end
elseif sign(u)<0
    if sign(yy)<0
        ans=q;
    else
        ans=1-q;
    end
end

