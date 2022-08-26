
function Data_extraction_Lantyer(stim_trace,resp_trace,Name)

    x=stim_trace(:,1);
    Stim=stim_trace(:,2);
    Stim=Stim*10^12;
    yyaxis left;
    plot(x,Stim);
    
    ylabel('Stimulus pA');
    yyaxis right;
    Resp=resp_trace(:,2);
    Resp=Resp*10^3
    plot(x,Resp);
    ylabel('Membrane Potential mV');
    mytitle='Plot Stim/Membrane ';
    mytitle=insertAfter(mytitle,"Membrane",Name),
    title( mytitle)
end




    