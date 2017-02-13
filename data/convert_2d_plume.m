function v_out  = convert_2d_plume( v_in )
% reshape vector for plotting
vsz = size(v_in); 
if (vsz(2) >1)
    v_out = reshape(v_in,128,320,7,[]);
else
    v_out = reshape(v_in,128,320,7);
end

end

