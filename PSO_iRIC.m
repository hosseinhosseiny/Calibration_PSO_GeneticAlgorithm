clc;
clear;
close all;
dat_mes = xlsread('dist.xlsx');
wse_mes = dat_mes (:,2)';
modelfile = 'GA_ANN';
net = importKerasNetwork('model_f.h5');

%% problem definition

CostFunction = @(x,y,z) RMSE(x,y,z) ; % Cost Function

nVar = 8 ;              %   Number of unknown varibles

VarSize = [1 nVar];     %   matrix size of unknown variales

VarMin = 0.001;           %   Lower bound of varibles
VarMax = 0.03;            %   Upper bound of varibles

%% Paramters of PSO

MaxIt = 100 ;           % Maximum Iteration

nPop = 100;              % Population Size (Swarm Size)
w = 1;                  % Inertia Coefficient
wdamp = 0.99 ;           % Damping Ratio of inertia weight
c1 = 2 ;                %Personal Acceleration Coefficient
c2 = 2 ;                %Social Acceleration Coefficient

%% Initialization
% The Particle Template
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Cost = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];

% Create Population Array
particle = repmat(empty_particle, nPop, 1);

% Initialize Global Best
GlobalBest.Cost = inf; % for minimizzation is inf and for maximization is -inf


% Initialize Population members
for i = 1:nPop
    
    % Generate Random Solution
    particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
    
    % Initialize Velocity
    particle(i).Velocity = zeros (VarSize);
    
    %Evaluation
    particle(i).Cost = CostFunction(particle(i).Position,net,wse_mes);
    
    %Update Personal Best
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.Cost = particle(i).Cost;
    
    % Update Global Best
    if particle(i).Best.Cost < GlobalBest.Cost
       GlobalBest = particle(i).Best;
    end
    
end

% Array to Hold Best Cost Value
BestCosts = zeros (MaxIt,1);


%% Main Loop of PSO

for it = 1:MaxIt
    
    for i =1:nPop
        
        % Update Velocity
        particle(i).Velocity = w * particle(i).Velocity + c1 * rand(VarSize) .* (particle(i).Best.Position - particle(i).Position)+c2 * rand(VarSize) .* (GlobalBest.Position - particle(i).Position);
          
          % Update Postion
          particle(i).Position = particle(i).Position + particle(i).Velocity;
          
          % Evaluation
          particle(i).Cost = CostFunction(particle(i).Position,net,wse_mes);
          
          % Update Personal Best
          if particle(i).Cost < particle(i).Best.Cost
              
              particle(i).Best.Position = particle(i).Position;
              particle(i).Best.Cost = particle(i).Cost;
              
               % Update Global Best
              if particle(i).Best.Cost < GlobalBest.Cost
                GlobalBest = particle(i).Best;
                %disp(num2str(GlobalBest.Cost))
              end
              
          end
          
    end
    
    % Store the Best Cost Value
    BestCosts(it) = GlobalBest.Cost;
    
    % Display Iteration Information
    disp([num2str(it) ' Best Cost = ' num2str(BestCosts(it)) ]);
    
    % Damping Inertia Weight 
    w = w * wdamp;
end

%% Results
figure;
%plot (BestCosts, 'LineWidth',2)
semilogy (BestCosts, 'LineWidth',2)
xlabel('Iteration')
ylabel('Best Cost')
grid on



