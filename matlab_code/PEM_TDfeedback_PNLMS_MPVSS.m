%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PEM_TDfeedback_VSS_MD_NLMS_new.m
% Example file for using the time-domain implementation of the
% PemAFC-based feedback cancellation algorithm
% Modified by Linh Tran based on Ann Spriet's code 
% Modified Practical variable step size (MPVSS)
% Author: Linh Tran
% Date: March 2016
% modified FA 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

%% Set Variables
prob_sig = 0;         % select among 0) without probe signal, 1)with probe signal as a white noise 
in_sig = 3;           % 0) white noise, 1) speech weighted noise, 2) real speech, 3) music

fs = 16000;          % sampling frequency
N = 80*fs;            % total number of samples

Kdb = 30;             % gain of forward path in dB
K = 10^(Kdb/20);        
d_k = 96;             % delay of the forward path K(q) in samples 
d_fb = 1;            % delay of the feedback cancellation path in samples
Lg_hat = 64;          % the full length of adaptive filter
SNR = 25;               % amplified signal to injected noise ratio

% Fixed Stepsize
% mu = 0.001;         % fixed step size
mu_c1 = zeros(N,1);
mu_c = zeros(N,1);
mu = zeros(N,1);

pe = zeros(N,1);
pm  = zeros(N,1);
p_vhatwh  = zeros(N,1);

%%%%%%%%%%%%%%%%%%%%%%%%%
%Settings Feedback path %
%%%%%%%%%%%%%%%%%%%%%%%%%
%% Feedback path 
load('mFBPathIRs16kHz_FF.mat');
E = mFBPathIRs16kHz_FF(:,3,1,1);
g = E - mean(E);  % feedback path and remove mean value
Lg = length(g);    % length of feedback path
Nfreq = 512;
G = fft(g,Nfreq);

load('mFBPathIRs16kHz_PhoneNear.mat');
Ec = mFBPathIRs16kHz_PhoneNear(:,3,1);
gc = Ec - mean(Ec);  % feedback path and remove mean value
Gc = fft(gc,Nfreq);

TDLy = zeros(Lg,1);            %time-delay vector true feedback path
   
%%%%%%%%%%%%%%%%%%%%%%%%%
%Settings desired signal%
%%%%%%%%%%%%%%%%%%%%%%%%%
%% Desired Signal (incoming signal)

 if in_sig == 0     
    % 0) incoming signal is a white noise
    Var_noise =0.001;
    input = sqrt(Var_noise)*randn(N,1);
 elseif in_sig == 1
    % 1) incoming signal is a synthesized speech
    Var = 1;
    h_den = [1;-2*0.96*cos(3000*2*pi/15750);0.96^2];
    v = sqrt(Var)*randn(N,1);     % v[k] is white noise with variance one
    input = filter(1,h_den,v);        % speech weighted noise 
 elseif in_sig == 2
    % 2) incoming signal is a real speech segment from NOIZEUS
%     load('HeadMid2_Speech_Vol095_0dgs_m1.mat')
%     u = HeadMid2_Speech_Vol095_0dgs_m1(1:N,1); % 50s
    
    load('HeadMid2_Speech_Vol095_0dgs_m1.mat')
    input1 = HeadMid2_Speech_Vol095_0dgs_m1(1:N,1);
    input = input1(8000:end);   

    
    
 else
    % 3) incoming signal is a music 
    load('HeadMid2_Music_Vol095_0dgs_m1');      
    input1 = HeadMid2_Music_Vol095_0dgs_m1;
    input = input1(16000:end);
%     load('Ext_music_G30dB_0dgs_m1.mat')
%     input = Ext_music_G30dB_0dgs_m1(1:N,1); % 80s
 end   
 
    input = input./max(abs(input));
    ff = fir1(64,[.025],'high');    
    u_ = filter(ff,1,input);  
    
u = zeros(N,1); 
for n = 1 : N
    % loop through input signal
    if n <= length(u_)
        u(n) = u_(n);
    else
        u(n) = u_(rem(n,length(u_))+1,1);
    end  
end 

%%%%%%%%%%%%%%%%%%%%%%%%%
%Settings Probe signal  %
%%%%%%%%%%%%%%%%%%%%%%%%%
%% Probe signal w(k)
    Var_P =0.001;
    if prob_sig == 1
        w=sqrt(Var_P)*randn(N,1);   % w[k] is a white noise;        
    else                            % With probe signal as a white noise        
        w=zeros(N,1);               % Without probe signal
    end
    
 % SNR
    if prob_sig == 1
        estKu = K * u;
        factor = SNRFactor(estKu,w,SNR);
        u = factor * u;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-whitening filter  %
%%%%%%%%%%%%%%%%%%%%%%%%%
     
La = 20;
framelength= 0.01*fs;
[AF,AR] = PemAFCinit_VSS(Lg_hat,La,framelength);
% [AF,AR] = PemAFCinit(Lg_hat,La,framelength);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%initialisation data vectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = zeros(N,1);         % loudspeaker signal
e_delay=zeros(N+d_k,1);
y_delayfb=zeros(N+d_fb,1);
m = zeros(N,1);         % received microphone signal
% ewh = zeros(N,1);
%%%%%%%%%%%%%%%%%%%%%%%%%
% PEM lattice algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%
% ewh(1)=0;
for k = 2:N 
    % Change the feedback path at sample N/2
    if k == N/2
       g = gc;       
       G = Gc; 
    end
    
     y(k) = K*e_delay(k) + w(k);

     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     % Simulated feedback path: computation of microphone signal
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
      TDLy = [y(k);TDLy(1:end-1,1)];
      m(k) = u(k) + g'*TDLy;        %received microphone signal
     
      y_delayfb(k+d_fb) = y(k);     
      
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %Feedback cancellation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For PEM-MPVSS: Linh's design
      [e_delay(k+d_k),mu_c1(k),mu_c(k),mu(k),AF,AR] = PemAFC_IPNLMS_MPVSS(m(k),y_delayfb(k),AF,AR,1);     

    
 %  Misaligment of the PEM-AFC
    g_hat = [zeros(d_fb,1);AF.gTD];
    G_hat = fft(g_hat,Nfreq);
    G_tilde = G(1:ceil(Nfreq/2))-G_hat(1:ceil(Nfreq/2));
    
  
    if mod(k/N*100, 2) == 0, 
        [num2str(k/N*100) '%'],
    end
end

    t = linspace(0,(N-1)/fs,N);
    
figure(4);plot(g);hold on;plot(g_hat,'r');grid on;hold off
figure(5);plot(t,y); grid on; 
xlabel('Time [s]');
ylabel('Amplitude');
legend ('PEM-IPNLMS-IPVSS');
