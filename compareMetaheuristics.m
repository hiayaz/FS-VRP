%% compareMetaheuristics.m
% Runs all five metaheuristics on the same FlowShop+VRP problem instance
% and plots a convergence comparison.
%
% USAGE
%   compareMetaheuristics          % runs with built-in demo problem
%   compareMetaheuristics(prob)    % runs with custom problem struct
%
% Each algorithm file must be on the MATLAB path:
%   runABCFlowShopVRP.m  (original)
%   runGWOFlowShopVRP.m
%   runWOAFlowShopVRP.m
%   runPSOFlowShopVRP.m
%   runSAFlowShopVRP.m
%   runTSFlowShopVRP.m
function compareMetaheuristics(problem,nJobs,mMachines,gridSize,i4)

    if nargin < 1 || isempty(problem)
        problem = buildDemoProblem();
    end
    rng("shuffle");
    % ---- Common parameters ----
    params.SN       = 50;
    params.maxCycle = 500;
    % params.seed     = 0;
    params.idleWeight = 1;
    params.VRPWeight  = 1;
    params.MSWeight   = 1;
    params.limit    = 100;   % ABC only

    % ---- Run each algorithm ----
    % fprintf('\n======================================\n');
    % fprintf('  FlowShop+VRP Metaheuristic Comparison\n');
    % fprintf('======================================\n');

    rABC = runABCFlowShopVRP(problem, params);
    rGWO = runGWOFlowShopVRP(problem, params);
    rWOA = runWOAFlowShopVRP(problem, params);
    rPSO = runPSOFlowShopVRP(problem, params);
    rSA  = runSAFlowShopVRP(problem,  params);
    rTS  = runTSFlowShopVRP(problem,  params);
    % ---- Summary table ----
    algorithms = {'ABC','GWO','WOA','PSO','SA','TS'};
    results    = {rABC, rGWO, rWOA, rPSO, rSA, rTS};
    folder  = 'C:\Users\User\OneDrive - erbakan.edu.tr\esra\maylab 2\Results';
    filename = sprintf('Machine%d--Job%d--Grid%d--Rep%d.mat', mMachines, nJobs, gridSize,i4);
    fullpath = fullfile(folder, filename);
    save(fullpath, 'results');

    % fprintf('\n========== SUMMARY TABLE ==========\n');
    % fprintf('%-6s | %-10s | %-10s | %-10s | %-10s\n', ...
    %         'Alg','Fitness','Makespan','VRPTravel','IdleTime');
    % fprintf('%s\n', repmat('-',1,58));
    % for k = 1:numel(algorithms)
    %     r = results{k};
    %     fprintf('%-6s | %-10.2f | %-10.2f | %-10.2f | %-10.2f\n', algorithms{k}, r.bestFitness, r.bestMakespan, r.bestVRPTravelTime, r.bestIdleTime);
    % end

    % ---- Convergence plot ----
    % figure('Name','Metaheuristic Convergence','NumberTitle','off');
    % colors = lines(numel(algorithms));
    % hold on;
    % for k = 1:numel(algorithms)
    %     plot(results{k}.history, 'LineWidth', 1.8, 'Color', colors(k,:), ...
    %          'DisplayName', algorithms{k});
    % end
    % hold off;
    % xlabel('Iteration / Cycle');
    % ylabel('Best Objective Value');
    % title('FlowShop+VRP – Convergence Comparison');
    % legend('Location','northeast');
    % grid on;
    % set(gca,'FontSize',12);
end

% -------------------------------------------------------------------
function problem = buildDemoProblem()
% 8-job, 3-machine flow-shop + VRP demo problem
    rng(0);
    nJobs    = 40;
    nMach    = 20;
    problem.P        = randi([1 10], nJobs, nMach);
    problem.coords   = rand(nJobs, 2) * 100;
    problem.weights  = randi([1 5], nJobs, 1);
    problem.capacity = 12;
    problem.depot    = [0 0];
    fprintf('Demo problem: %d jobs, %d machines, capacity=%d\n', ...
            nJobs, nMach, problem.capacity);
end
