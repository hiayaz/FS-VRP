function result = runWOAFlowShopVRP(problem, params)
tic
% runWOAFlowShopVRP
% Whale Optimization Algorithm (WOA) for integrated flow-shop + VRP.
%
% WOA MECHANISMS (adapted for permutations)
%   1. Shrinking encircling (exploitation): borrow segment from best whale
%   2. Spiral bubble-net (exploitation): cyclic segment rotation
%   3. Random search (exploration): move toward a random whale
%
% OBJECTIVE
%   Minimize: deliveryFinish + idleWeight * totalMachineIdleTime
%
% INPUT  / OUTPUT  -- same as ABC version

    if nargin < 2 || isempty(params), params = struct(); end
    params  = setDefaultParams(params);
    problem = validateProblem(problem);

    if ~isempty(params.seed), rng(params.seed); else, rng('shuffle'); end

    nJobs = size(problem.P, 1);
    SN    = params.SN;

    % ---- Initialize ----
    whales = repmat(emptySolution(), SN, 1);
    for s = 1:SN
        whales(s) = evaluateSequence(randperm(nJobs), problem, params.idleWeight);
    end

    [~, bIdx] = min([whales.fitness]);
    best = whales(bIdx);
    history = zeros(params.maxCycle, 1);

    % ---- WOA main loop ----
    for cycle = 1:params.maxCycle
        a  = 2 - 2 * (cycle - 1) / (params.maxCycle - 1);  % 2 -> 0
        a2 = -1 - (cycle - 1) / (params.maxCycle - 1);       % -1 -> -2

        for s = 1:SN
            A = 2 * a * rand - a;
            p = rand;

            if p < 0.5
                if abs(A) < 1
                    % Shrinking encircling: borrow from best
                    candidateSeq = encircleMove(whales(s).sequence, best.sequence, a);
                else
                    % Exploration: move toward random whale
                    randIdx = randi(SN);
                    candidateSeq = encircleMove(whales(s).sequence, whales(randIdx).sequence, a);
                end
            else
                % Spiral bubble-net update
                candidateSeq = spiralMove(whales(s).sequence, best.sequence);
            end

            candidate = evaluateSequence(candidateSeq, problem, params.idleWeight);
            if candidate.fitness < whales(s).fitness
                whales(s) = candidate;
                if candidate.fitness < best.fitness
                    best = candidate;
                end
            end
        end

        history(cycle) = best.fitness;
    end
    WOATime=toc;
    result = buildResult(best, history, params,WOATime);
    % fprintf('\n=== Whale Optimization Algorithm ===\n');
    % printResult(result);
end

% -------------------------------------------------------------------
function newSeq = encircleMove(currentSeq, targetSeq, a)
% Borrow a segment from targetSeq; proportion scales with a
    n        = numel(currentSeq);
    segLen   = max(1, round((2 - a) / 2 * randi([1, floor(n/2)])));
    segLen   = min(segLen, n);
    startPos = randi(n - segLen + 1);
    segment  = targetSeq(startPos : startPos + segLen - 1);
    remaining = currentSeq(~ismember(currentSeq, segment));
    insertAt  = randi(numel(remaining) + 1);
    newSeq   = [remaining(1:insertAt-1), segment, remaining(insertAt:end)];
end

function newSeq = spiralMove(currentSeq, bestSeq)
% Cyclic rotation of a random segment toward best's segment ordering
    n = numel(currentSeq);
    b = 1;               % spiral shape constant
    l = (rand - 0.5) * 2; % l in [-1, 1]
    segLen   = max(1, randi(floor(n/2)));
    startPos = randi(n - segLen + 1);
    segment  = bestSeq(startPos : startPos + segLen - 1);

    % Rotate segment by l-inspired offset
    offset   = round(abs(l) * segLen);
    rotated  = [segment(offset+1:end), segment(1:offset)];

    remaining = currentSeq(~ismember(currentSeq, rotated));
    insertAt  = randi(numel(remaining) + 1);
    newSeq   = [remaining(1:insertAt-1), rotated, remaining(insertAt:end)];
end

% ===== Shared helper functions =====
function params = setDefaultParams(params)
    if ~isfield(params, 'SN') || isempty(params.SN),               params.SN = 30;         end
    if ~isfield(params, 'limit') || isempty(params.limit),         params.limit = 100;      end
    if ~isfield(params, 'maxCycle') || isempty(params.maxCycle),   params.maxCycle = 200;   end
    if ~isfield(params, 'seed'),                                    params.seed = [];        end
    if ~isfield(params, 'idleWeight') || isempty(params.idleWeight), params.idleWeight = 1; end

    if params.SN < 1 || params.maxCycle < 1
        error('params.SN and params.maxCycle must be positive.');
    end
    if params.idleWeight < 0
        error('params.idleWeight must be nonnegative.');
    end
end

function problem = validateProblem(problem)
    required = {'P', 'coords', 'weights', 'capacity'};
    for k = 1:numel(required)
        if ~isfield(problem, required{k})
            error('Problem field "%s" is missing.', required{k});
        end
    end
    if ~isfield(problem, 'depot') || isempty(problem.depot)
        problem.depot = [0 0];
    end
    nJobs = size(problem.P, 1);
    assert(nJobs >= 1,                   'problem.P must have at least one job.');
    assert(size(problem.P, 2) >= 1,      'problem.P must have at least one machine.');
    assert(size(problem.coords,1)==nJobs && size(problem.coords,2)==2, ...
                                         'problem.coords must be nJobs x 2.');
    assert(numel(problem.weights)==nJobs,'problem.weights must have one entry per job.');
    problem.weights = problem.weights(:);
    assert(all(problem.weights > 0),     'All job weights must be positive.');
    assert(problem.capacity >= max(problem.weights), ...
                                         'Vehicle capacity too small.');
    assert(all(problem.P(:) >= 0),       'Processing times must be nonneg.');
end

function sol = emptySolution()
    sol = struct('sequence',[],'fitness',inf,'makespan',inf,'idleTime',inf, ...
                 'idleByMachine',[],'vrpTravelTime',inf,'deliveryFinish',inf, ...
                 'completionTimes',[],'routes',[],'trial',0);
end

function sol = evaluateSequence(seq, problem, idleWeight)
    [makespan, C, idleTime, idleByMachine] = flowShopCompletion(seq, problem.P);
    [vrpTravelTime, routes]                = evaluateVRPMovementOnly(seq, problem);
    deliveryFinish = makespan + vrpTravelTime;
    fitness        = deliveryFinish + idleWeight * idleTime;

    sol                 = emptySolution();
    sol.sequence        = seq;
    sol.fitness         = fitness;
    sol.makespan        = makespan;
    sol.idleTime        = idleTime;
    sol.idleByMachine   = idleByMachine;
    sol.vrpTravelTime   = vrpTravelTime;
    sol.deliveryFinish  = deliveryFinish;
    sol.completionTimes = C;
    sol.routes          = routes;
end

function [makespan, C, totalIdle, idleByMachine] = flowShopCompletion(seq, P)
    n = numel(seq);
    m = size(P, 2);
    C = zeros(n, m);
    for i = 1:n
        job = seq(i);
        for machine = 1:m
            if i == 1 && machine == 1
                C(i,machine) = P(job,machine);
            elseif i == 1
                C(i,machine) = C(i,machine-1) + P(job,machine);
            elseif machine == 1
                C(i,machine) = C(i-1,machine) + P(job,machine);
            else
                C(i,machine) = max(C(i-1,machine), C(i,machine-1)) + P(job,machine);
            end
        end
    end
    makespan     = C(end,end);
    busyByMachine = sum(P(:,:),1);
    idleByMachine = C(end,:) - C(1,:) - busyByMachine + P(seq(1),:);
    totalIdle    = sum(idleByMachine);
end

function [totalTravelTime, routes] = evaluateVRPMovementOnly(seq, problem)
    batches = buildCapacityBatches(seq, problem.weights, problem.capacity);
    nBatches = numel(batches);
    routes = repmat(struct('jobs',[],'routeOrder',[],'travelTime',0), nBatches, 1);
    totalTravelTime = 0;
    for b = 1:nBatches
        batchJobs   = batches{b};
        batchCoords = problem.coords(batchJobs, :);
        [routeTravelTime, localRouteOrder] = solveTSP(batchCoords, problem.depot);
        routes(b).jobs       = batchJobs;
        routes(b).routeOrder = batchJobs(localRouteOrder);
        routes(b).travelTime = routeTravelTime;
        totalTravelTime      = totalTravelTime + routeTravelTime;
    end
end

function batches = buildCapacityBatches(seq, weights, capacity)
    batches      = {};
    currentBatch = [];
    currentLoad  = 0;
    for k = 1:numel(seq)
        job = seq(k);  w = weights(job);
        if currentLoad + w <= capacity
            currentBatch = [currentBatch, job]; currentLoad = currentLoad + w;
        else
            batches{end+1} = currentBatch;
            currentBatch = job; currentLoad = w;
        end
    end
    if ~isempty(currentBatch), batches{end+1} = currentBatch; end
end

function [bestDistance, bestOrder] = solveTSP(customerCoords, depot)
    n = size(customerCoords,1);
    if     n == 0, bestDistance = 0; bestOrder = [];
    elseif n == 1, bestDistance = 2*manhattanDistance(depot,customerCoords(1,:)); bestOrder = 1;
    elseif n <= 8, [bestDistance,bestOrder] = exactTSP(customerCoords,depot);
    else,          [bestDistance,bestOrder] = heuristicTSP(customerCoords,depot);
    end
end

function [bestDistance, bestOrder] = exactTSP(customerCoords, depot)
    n = size(customerCoords,1);
    allOrders = perms(1:n); bestDistance = inf; bestOrder = [];
    for r = 1:size(allOrders,1)
        d = routeDistance(allOrders(r,:), customerCoords, depot);
        if d < bestDistance, bestDistance = d; bestOrder = allOrders(r,:); end
    end
end

function [bestDistance, bestOrder] = heuristicTSP(customerCoords, depot)
    n         = size(customerCoords,1);
    D         = distanceMatrix([depot; customerCoords]);
    unvisited = true(1,n); order = zeros(1,n); currentNode = 1;
    for k = 1:n
        cand = D(currentNode, 2:end); cand(~unvisited) = inf;
        [~, next] = min(cand); order(k) = next;
        unvisited(next) = false; currentNode = next + 1;
    end
    bestOrder    = twoOptImprove(order, customerCoords, depot);
    bestDistance = routeDistance(bestOrder, customerCoords, depot);
end

function improvedOrder = twoOptImprove(order, coords, depot)
    improvedOrder = order; bestDistance = routeDistance(improvedOrder,coords,depot);
    improved = true;
    while improved
        improved = false; n = numel(improvedOrder);
        for i = 1:n-1
            for j = i+1:n
                candidate = improvedOrder; candidate(i:j) = candidate(j:-1:i);
                d = routeDistance(candidate,coords,depot);
                if d < bestDistance, improvedOrder = candidate; bestDistance = d; improved = true; end
            end
        end
    end
end

function d = routeDistance(order, coords, depot)
    d = manhattanDistance(depot, coords(order(1),:));
    for k = 1:numel(order)-1
        d = d + manhattanDistance(coords(order(k),:), coords(order(k+1),:));
    end
    d = d + manhattanDistance(coords(order(end),:), depot);
end

function D = distanceMatrix(points)
    n = size(points,1); D = zeros(n,n);
    for i = 1:n
        for j = i+1:n
            D(i,j) = manhattanDistance(points(i,:),points(j,:)); D(j,i) = D(i,j);
        end
    end
end

function d = manhattanDistance(a,b), d = sum(abs(a-b)); end

function neigh = makeNeighbor(seq)
    n = numel(seq); neigh = seq; if n < 2, return; end
    switch randi(3)
        case 1  % swap
            idx = randperm(n,2); neigh(idx) = neigh(fliplr(idx));
        case 2  % insertion
            idx = randperm(n,2); i = idx(1); j = idx(2);
            job = neigh(i); neigh(i) = [];
            if j > numel(neigh)+1, j = numel(neigh)+1; end
            neigh = [neigh(1:j-1), job, neigh(j:end)];
        case 3  % inversion
            idx = sort(randperm(n,2)); neigh(idx(1):idx(2)) = neigh(idx(2):-1:idx(1));
    end
end

function result = buildResult(best, history, params,WOATime)
    result.bestSequence       = best.sequence;
    result.bestFitness        = best.fitness;
    result.bestMakespan       = best.makespan;
    result.bestIdleTime       = best.idleTime;
    result.bestIdleByMachine  = best.idleByMachine;
    result.bestVRPTravelTime  = best.vrpTravelTime;
    result.bestDeliveryFinish = best.deliveryFinish;
    result.bestRoutes         = best.routes;
    result.history            = history;
    result.params             = params;
    result.time               = WOATime;
end

function printResult(result)
    fprintf('Best objective value   : %.4f', result.bestFitness);
    fprintf('  Delivery finish      : %.4f', result.bestDeliveryFinish);
    fprintf('  Production makespan  : %.4f', result.bestMakespan);
    fprintf('  VRP travel time      : %.4f', result.bestVRPTravelTime);
    fprintf('  Total machine idle   : %.4f', result.bestIdleTime);
    fprintf('Idle time by machine:');
    for m = 1:numel(result.bestIdleByMachine)
        fprintf('  Machine %d: %.4f', m, result.bestIdleByMachine(m));
    end
    fprintf('Best sequence: '); disp(result.bestSequence);
    fprintf('Routes:');
    for r = 1:numel(result.bestRoutes)
        fprintf('  Route %d | travel %.2f | jobs: ', r, result.bestRoutes(r).travelTime);
        disp(result.bestRoutes(r).routeOrder);
    end
end
