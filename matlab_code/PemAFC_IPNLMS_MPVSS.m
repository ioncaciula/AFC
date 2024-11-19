% function [e,AF,AR] = PemAFC_IPNLMS_MPVSS(Mic,Ls,AF,AR,UpdateFC)
function [e,mu_c1,mu_c,mu,AF,AR] = PemAFC_IPNLMS_MPVSS(Mic,Ls,AF,AR,UpdateFC)
%
%function [e,AF,AR] = PemAFC(Mic,Ls,AF,AR,UpdateFC,RemoveDC)
%  
%Update equation of time-domain LMS-based implementation of PEM-based
%adaptive feedback canceller
%
%INPUTS:
% * Mic = microphone sample
% * Ls  = loudspeaker sample
% * AF  = time-domain LMS-based feedback canceller and its properties
%          -AF.wTD: time-domain filter coefficients (dimensions: AF.N x 1)  
%          -AF.N  : time-domain filter length
%          -AF.mu : stepsize
%          -AF.p  : power in step-size normalization
%          -AF.lambda: weighing factor for power computation
%          -AF.TDLLs: time-delay line of loudspeaker samples
%          (dimensions: AF.N x 1)
%          -AF.TDLLswh: time-delay line of pre-whitened loudspeaker
%          samples (dimensions: AF.N x 1)
%
% * AR        = auto-regressive model and its properties
%                -AR.w           : coefficients of previous AR-model
%                -AR.N           : filter length AR model (Note: N=Nh+1)
%                -AR.framelength : framelength on which AR model is estimated
%                -AR.TDLMicdelay : time-delay line of microphone samples
%                                  (dimensions: AR.framelength+1 x 1)
%                -AR.TDLLsdelay  : time-delay line of loudspeaker samples
%                                  (dimensions: AR.framelength+1 x 1)
%                -AR.TDLMicwh    : time-delay line of pre-whitened
%                                  microphone signal
%                                  (dimensions: AR.N x 1)
%                -AR.TDLLswh     : time-delay line of pre-whitened
%                                  loudspeaker signal
%                                  (dimensions: AR.N x 1)
%                -AR.frame       : frame of AR.framelength error signals
%                                  on which AR model is computed  
%                -AR.frameindex
% * UpdateFC = boolean that indicates whether or not the feedback canceller should be updated 
%                    (1 = update feedback canceller; 0 = do not update feedback canceller)  
% * RemoveDC   = boolean that indicates whether or not the DC component of the estimated feedback path should be removed 
%                    (1 = remove DC of feedback canceller; 0 = do not remove DC of feedback canceller)  
%OUTPUTS:
% * e          = feedback-compensated signal 
% * AR         = updated AR-model and its properties
% * AF         = updated feedback canceller and its properties 
%
%
%
%Date: December, 2007  
%Copyright: (c) 2007 by Ann Spriet
%e-mail: ann.spriet@esat.kuleuven.be

% Modified Practical variable step size (MPVSS)
% Author: Linh Tran
% Date: March 2016

delta = 1e-10;
AF.TDLLs = [Ls;AF.TDLLs(1:end-1,1)];
e     = Mic - AF.gTD'*AF.TDLLs;
e = 2*tanh(0.5*e);

%Delay microphone and loudspeaker signal by framelength
[Micdelay,AR.TDLMicdelay]=DelaySample(Mic,AR.framelength,AR.TDLMicdelay); 
[Lsdelay,AR.TDLLsdelay]=DelaySample(Ls,AR.framelength,AR.TDLLsdelay);

% Filter microphone and loudspeaker signal with AR-model
[Micwh,AR.TDLMicwh] = FilterSample(Micdelay,AR.w,AR.TDLMicwh);
[Lswh,AR.TDLLswh]   = FilterSample(Lsdelay,AR.w,AR.TDLLswh);

%Update AR-model  
AR.frame=[e;AR.frame(1:AR.framelength-1)];
     
if and(AR.frameindex==AR.framelength-1,AR.N-1>0)
  R=zeros(AR.N,1);
  for j= 1:AR.N
     R(j,1) = (AR.frame'*[AR.frame(j:length(AR.frame)); zeros(j-1,1)])/AR.framelength;
  end
  [a,Ep] = levinson(R,AR.N-1);
  AR.w=a';
end

AR.frameindex=AR.frameindex+1;
      
if AR.frameindex==AR.framelength
  AR.frameindex=0;
end

AF.TDLLswh = [Lswh;AF.TDLLswh(1:end-1,1)];
vhatwh = AF.gTD'*AF.TDLLswh;
ep = Micwh - vhatwh;

if UpdateFC == 1
    %% Compute VSS-NLMS_new
    
    xi = 1e-6;
    lamda1 = 0.9999;

    mu_max = 0.01;
    mu_min = 0.001;

    % Compute recursively estimated variances p_m and p_e (redo)
      AF.p_m = lamda1*AF.p_m + (1-lamda1)*(Micwh^2);
      AF.p_e = lamda1*AF.p_e + (1-lamda1)*(ep^2);
      AF.p_vhatwh = lamda1*AF.p_vhatwh + (1-lamda1)*(vhatwh^2);
      
      p_temp = abs(AF.p_m - AF.p_vhatwh);
      mu_c1 = abs(1-sqrt(p_temp)/(sqrt(AF.p_e)+xi));
      mu_c = mu_max*mu_c1;

      mu_c3 = min(mu_max,mu_c); % original
      mu = max(mu_min,mu_c3);   % original
      

%% l1-norm IPNLMS proposed by Benesty and Gay, 2002 
% mu = 0.01;
aa = 0; 
kd = (1-aa)/(2*length(AF.gTD)) + (1+aa)*abs(AF.gTD)/(delta + 2*sum(abs(AF.gTD)));
Kd = diag(kd);
AF.gTD = AF.gTD + (mu/(AF.TDLLswh'*Kd*AF.TDLLswh + delta*(1-aa)/(2*length(AF.gTD))))*Kd*AF.TDLLswh.*ep;
%%
% PEM-NLMS
% AF.gTD = AF.gTD +(mu/(norm(AF.TDLLswh)^2+delta))*AF.TDLLswh.*ep;
      
% Remove DC
AF.gTD = AF.gTD - mean(AF.gTD);

end