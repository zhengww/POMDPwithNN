
% read data from IRC and neural networks
TX = readtable('neural_decoding_X.csv');
X = TX{:,:}; % belief and location data

Ta = readtable('neural_decoding_y.csv');
a = Ta{:,:};

Tpih = readtable('neural_decoding_POMDPpolicy.csv');
pih = Tpih{2:end,:};

T = length(a);


b = X(:,1:2); % beliefs on two boxes
L = X(:,3)+1; % location
nL = 3;


% introduce nonlinear features for regression.

nonlinearity = 2;

if (nonlinearity == 1)
    
    disp('Using indicator nonlinearity.');
    
    % Indicator function on quantized beliefs.
    
    b = X(:,1:2); % beliefs on two boxes
    L = X(:,3)+1; % location
    nL = 3;

    dq = 0.5; % spacing of bins
    bq = round( b/dq - .5 ); % quantized beliefs
    bi = bq - min(bq,[],'all') + 1;
    nbq = length(unique(bq));

    biqK = zeros(T,nbq,nbq,nL); % multidimensional indicator function on quantized beliefs
    for t = 1:T
        biqK( t, bi(t,1), bi(t,2), L(t) ) = 1; % bivariate subscript to linear index
    end
    biq1 = reshape(biqK,T,[]);

    f = biq1(1:Tdata,:);
    
elseif (nonlinearity == 2)
    
	disp('Using RBF nonlinearity.');

    
    % Radial Basis Functions
    dq = .2; % RBF spacing
    sigma2 = (.5*dq)^2; % length scale
    
    % define centers
    q1L = dq/2 : dq : (1-dq/2); % centers on each axis
    nq1 = length(q1L);
    nq = nq1^2;
   
    qL = zeros(nq,2);
    ind = 0;
    for qi = q1L
        for qj = q1L
            ind = ind + 1;
            qL(ind,:) = [qi,qj];
        end
    end
        
    b_RBF = zeros(T,nq,nL);
    for t=1:T
        for ind = 1:nq
            b_RBF(t,ind,L(t)) = exp( -( (b(t,1)-qL(ind,1))^2 + (b(t,2)-qL(ind,2))^2 )/(2*sigma2) );
        end
    end
    
    
    f = reshape(b_RBF,T,[]);
    
else % linear

	disp('Using linear features.');

    f = X;

end    

    


% indicator function for actions (for mnrfit)
na = size(pih,2);
ai = zeros(T,na);
for t=1:T
    ai(t,a(t)+1) = 1;
end

Tdata = 20000; % use less data
fsource = f(1:Tdata,:);
%atarget = ai(1:Tdata,:);
atarget = a(1:Tdata)  +1;
pihs = pih(1:Tdata,:);


% multinomial regression -------------------------------
B = mnrfit(fsource,atarget);


%pic = mnrval(B,fsource);


% cross validate
Ttest = 1000;
ftest = f( (Tdata+1) : (Tdata+Ttest), : );
pic_test = mnrval(B,ftest);
pih_test = pih((Tdata+1) : (Tdata+Ttest), :);


plot(pih_test,pic_test,'k.');
axis square;
axis([0 1 0 1]);

figure; 
hold;
for i = 1:5
    plot(pih_test(:, i),pic_test(:, i), '.');
end
axis square;
axis([0 1 0 1]);


