function result = runABCFlowShopVRP(problem, params)
tic
% runABCFlowShopVRP
% Clean ABC algorithm for an integrated permutation flow-shop +
% single-vehicle capacitated routing problem with machine idle time.
%
% UPDATED DELIVERY LOGIC
%   VRP time is calculated ONLY as the sum of vehicle movement times.
%   Waiting times, release times, and vehicle idle times are NOT added to
%   VRPTravelTime.
%
% OBJECTIVE
%   Minimize:
%       deliveryFinish + idleWeight * totalMachineIdleTime
%
%   where:
%       VRPTravelTime  = sum of all route travel distances/times
%       deliveryFinish = makespan + VRPTravelTime
%
% INPUT
%   problem.P         : nJobs x mMachines processing-time matrix
%   problem.coords    : nJobs x 2 customer coordinates
%   problem.weights   : nJobs x 1 job demands/weights
%   problem.capacity  : scalar vehicle capacity
%   problem.depot     : 1 x 2 depot coordinate, optional, default [0 0]
%
%   params.SN         : number of food sources, default 30
%   params.limit      : abandonment limit, default 100
%   params.maxCycle   : maximum ABC cycles, default 200
%   params.seed       : random seed, optional, default []
%   params.idleWeight : coefficient of idle time in objective, default 1
%
% OUTPUT
%   result.bestSequence
%   result.bestFitness
%   result.bestMakespan
%   result.bestIdleTime
%   result.bestIdleByMachine
%   result.bestVRPTravelTime
%   result.bestDeliveryFinish
%   result.bestRoutes
%   result.history

    if nargin < 2 || isempty(params)
        params = struct();
    end

    params = setDefaultParams(params);
    problem = validateProblem(problem);

    if ~isempty(params.seed)
        rng(params.seed);
    else
        rng('shuffle');
    end

    nJobs = size(problem.P, 1);

    % ---------- Initial population ----------
    sol = repmat(emptySolution(), params.SN, 1);
    for s = 1:params.SN
        seq = randperm(nJobs);
        sol(s) = evaluateSequence(seq, problem, params.idleWeight);
    end

    [~, bestIdx] = min([sol.fitness]);
    best = sol(bestIdx);
    history = zeros(params.maxCycle, 1);

    % ---------- ABC main loop ----------
    for cycle = 1:params.maxCycle

        % Employed bees
        for s = 1:params.SN
            candidateSeq = makeNeighbor(sol(s).sequence);
            candidate = evaluateSequence(candidateSeq, problem, params.idleWeight);
            sol(s) = greedySelection(sol(s), candidate);
        end

        % Onlooker bees
        probs = selectionProbabilities([sol.fitness]);
        selected = 0;
        s = 1;
        while selected < params.SN
            if rand <= probs(s)
                selected = selected + 1;
                candidateSeq = makeNeighbor(sol(s).sequence);
                candidate = evaluateSequence(candidateSeq, problem, params.idleWeight);
                sol(s) = greedySelection(sol(s), candidate);
            end
            s = mod(s, params.SN) + 1;
        end

        % Scout bees
        for s = 1:params.SN
            if sol(s).trial >= params.limit
                sol(s) = evaluateSequence(randperm(nJobs), problem, params.idleWeight);
            end
        end

        % Best update
        [~, currentBestIdx] = min([sol.fitness]);
        if sol(currentBestIdx).fitness < best.fitness
            best = sol(currentBestIdx);
        end
        history(cycle) = best.fitness;
    end

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
    result.time               = toc;
    % fprintf('\n=== Artificial Bee Colony ===\n');
    % printResult(result);
end

% =======================================================================
% Local functions
% =======================================================================

function params = setDefaultParams(params)
    if ~isfield(params, 'SN') || isempty(params.SN)
        params.SN = 30;
    end
    if ~isfield(params, 'limit') || isempty(params.limit)
        params.limit = 100;
    end
    if ~isfield(params, 'maxCycle') || isempty(params.maxCycle)
        params.maxCycle = 200;
    end
    if ~isfield(params, 'seed')
        params.seed = [];
    end
    if ~isfield(params, 'idleWeight') || isempty(params.idleWeight)
        params.idleWeight = 1;
    end

    if params.SN < 1 || params.limit < 1 || params.maxCycle < 1
        error('params.SN, params.limit and params.maxCycle must be positive.');
    end
    if params.idleWeight < 0
        error('params.idleWeight must be nonnegative.');
    end
end

function problem = validateProblem(problem)
    requiredFields = {'P', 'coords', 'weights', 'capacity'};
    for k = 1:numel(requiredFields)
        if ~isfield(problem, requiredFields{k})
            error('Problem field "%s" is missing.', requiredFields{k});
        end
    end

    if ~isfield(problem, 'depot') || isempty(problem.depot)
        problem.depot = [0 0];
    end

    nJobs = size(problem.P, 1);

    if nJobs < 1
        error('problem.P must contain at least one job.');
    end
    if size(problem.P, 2) < 1
        error('problem.P must contain at least one machine.');
    end
    if size(problem.coords, 1) ~= nJobs || size(problem.coords, 2) ~= 2
        error('problem.coords must be nJobs x 2.');
    end
    if numel(problem.weights) ~= nJobs
        error('problem.weights must contain one weight for each job.');
    end

    problem.weights = problem.weights(:);

    if any(problem.weights <= 0)
        error('All job weights must be positive.');
    end
    if problem.capacity < max(problem.weights)
        error('Vehicle capacity must be at least as large as the largest job weight.');
    end
    if any(problem.P(:) < 0)
        error('Processing times must be nonnegative.');
    end
end

function sol = emptySolution()
    sol = struct( ...
        'sequence', [], ...
        'fitness', inf, ...
        'makespan', inf, ...
        'idleTime', inf, ...
        'idleByMachine', [], ...
        'vrpTravelTime', inf, ...
        'deliveryFinish', inf, ...
        'completionTimes', [], ...
        'routes', [], ...
        'trial', 0);
end

function sol = evaluateSequence(seq, problem, idleWeight)
    [makespan, C, idleTime, idleByMachine] = flowShopCompletion(seq, problem.P);
    [vrpTravelTime, routes] = evaluateVRPMovementOnly(seq, problem);

    % Delivery finish no longer includes waiting times.
    % It includes production completion plus only vehicle movement time.
    deliveryFinish = makespan + vrpTravelTime;

    fitness = deliveryFinish + idleWeight * idleTime;

    sol = emptySolution();
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
                C(i, machine) = P(job, machine);
            elseif i == 1
                C(i, machine) = C(i, machine - 1) + P(job, machine);
            elseif machine == 1
                C(i, machine) = C(i - 1, machine) + P(job, machine);
            else
                C(i, machine) = max(C(i - 1, machine), C(i, machine - 1)) + P(job, machine);
            end
        end
    end

    makespan = C(end, end);

    % Machine idle time over production horizon [0, makespan].
    busyByMachine = sum(P(:,:),1);
    idleByMachine = C(end,:) - C(1,:)-busyByMachine+P(seq(1),:);
    totalIdle = sum(idleByMachine);
end

function [totalTravelTime, routes] = evaluateVRPMovementOnly(seq, problem)
    batches = buildCapacityBatches(seq, problem.weights, problem.capacity);
    nBatches = numel(batches);

    routes = repmat(struct( ...
        'jobs', [], ...
        'routeOrder', [], ...
        'travelTime', 0), nBatches, 1);

    totalTravelTime = 0;

    for b = 1:nBatches
        batchJobs = batches{b};
        batchCoords = problem.coords(batchJobs, :);
        [routeTravelTime, localRouteOrder] = solveTSP(batchCoords, problem.depot);

        routes(b).jobs        = batchJobs;
        routes(b).routeOrder  = batchJobs(localRouteOrder);
        routes(b).travelTime  = routeTravelTime;

        totalTravelTime = totalTravelTime + routeTravelTime;
    end
end

function batches = buildCapacityBatches(seq, weights, capacity)
    batches = {};
    currentBatch = [];
    currentLoad = 0;

    for k = 1:numel(seq)
        job = seq(k);
        w = weights(job);

        if currentLoad + w <= capacity
            currentBatch = [currentBatch, job]; %#ok<AGROW>
            currentLoad = currentLoad + w;
        else
            batches{end + 1} = currentBatch; %#ok<AGROW>
            currentBatch = job;
            currentLoad = w;
        end
    end

    if ~isempty(currentBatch)
        batches{end + 1} = currentBatch;
    end
end

function [bestDistance, bestOrder] = solveTSP(customerCoords, depot)
    n = size(customerCoords, 1);

    if n == 0
        bestDistance = 0;
        bestOrder = [];
    elseif n == 1
        bestDistance = 2 * manhattanDistance(depot, customerCoords(1, :));
        bestOrder = 1;
    elseif n <= 8
        [bestDistance, bestOrder] = exactTSP(customerCoords, depot);
    else
        [bestDistance, bestOrder] = heuristicTSP(customerCoords, depot);
    end
end

function [bestDistance, bestOrder] = exactTSP(customerCoords, depot)
    n = size(customerCoords, 1);
    allOrders = perms(1:n);
    bestDistance = inf;
    bestOrder = [];

    for r = 1:size(allOrders, 1)
        order = allOrders(r, :);
        d = routeDistance(order, customerCoords, depot);
        if d < bestDistance
            bestDistance = d;
            bestOrder = order;
        end
    end
end

function [bestDistance, bestOrder] = heuristicTSP(customerCoords, depot)
    n = size(customerCoords, 1);
    D = distanceMatrix([depot; customerCoords]);

    unvisited = true(1, n);
    order = zeros(1, n);
    currentNode = 1;

    for k = 1:n
        candidateDistances = D(currentNode, 2:end);
        candidateDistances(~unvisited) = inf;
        [~, nextCustomer] = min(candidateDistances);
        order(k) = nextCustomer;
        unvisited(nextCustomer) = false;
        currentNode = nextCustomer + 1;
    end

    bestOrder = twoOptImprove(order, customerCoords, depot);
    bestDistance = routeDistance(bestOrder, customerCoords, depot);
end

function improvedOrder = twoOptImprove(order, coords, depot)
    improvedOrder = order;
    bestDistance = routeDistance(improvedOrder, coords, depot);
    improved = true;

    while improved
        improved = false;
        n = numel(improvedOrder);
        for i = 1:n-1
            for j = i+1:n
                candidate = improvedOrder;
                candidate(i:j) = candidate(j:-1:i);
                candidateDistance = routeDistance(candidate, coords, depot);
                if candidateDistance < bestDistance
                    improvedOrder = candidate;
                    bestDistance = candidateDistance;
                    improved = true;
                end
            end
        end
    end
end

function d = routeDistance(order, coords, depot)
    d = manhattanDistance(depot, coords(order(1), :));
    for k = 1:numel(order)-1
        d = d + manhattanDistance(coords(order(k), :), coords(order(k+1), :));
    end
    d = d + manhattanDistance(coords(order(end), :), depot);
end

function D = distanceMatrix(points)
    n = size(points, 1);
    D = zeros(n, n);
    for i = 1:n
        for j = i+1:n
            D(i, j) = manhattanDistance(points(i, :), points(j, :));
            D(j, i) = D(i, j);
        end
    end
end

function d = manhattanDistance(a, b)
    d = sum(abs(a - b));
end

function neigh = makeNeighbor(seq)
    n = numel(seq);
    neigh = seq;

    if n < 2
        return;
    end

    moveType = randi(3);

    switch moveType
        case 1 % swap
            idx = randperm(n, 2);
            neigh(idx) = neigh(fliplr(idx));

        case 2 % insertion
            idx = randperm(n, 2);
            i = idx(1);
            j = idx(2);
            job = neigh(i);
            neigh(i) = [];
            if j > numel(neigh) + 1
                j = numel(neigh) + 1;
            end
            neigh = [neigh(1:j-1), job, neigh(j:end)];

        case 3 % inversion
            idx = sort(randperm(n, 2));
            neigh(idx(1):idx(2)) = neigh(idx(2):-1:idx(1));
    end
end

function selected = greedySelection(current, candidate)
    if candidate.fitness < current.fitness
        selected = candidate;
        selected.trial = 0;
    else
        selected = current;
        selected.trial = current.trial + 1;
    end
end

function probs = selectionProbabilities(fitnessValues)
    shifted = fitnessValues - min(fitnessValues) + eps;
    quality = 1 ./ shifted;
    probs = quality ./ sum(quality);
end

function printResult(result)
    fprintf('\nBest objective value: %.4f\n', result.bestFitness);
    fprintf('  Delivery finish = makespan + pure VRP travel time\n');
    fprintf('  Delivery finish           : %.4f\n', result.bestDeliveryFinish);
    fprintf('  Production makespan       : %.4f\n', result.bestMakespan);
    fprintf('  Pure VRP travel time      : %.4f\n', result.bestVRPTravelTime);
    fprintf('  Total machine idle time   : %.4f\n', result.bestIdleTime);
    fprintf('  Idle weight in objective  : %.4f\n', result.params.idleWeight);

    fprintf('\nIdle time by machine:\n');
    for m = 1:numel(result.bestIdleByMachine)
        fprintf('  Machine %d idle time: %.4f\n', m, result.bestIdleByMachine(m));
    end

    fprintf('\nBest sequence:\n');
    disp(result.bestSequence);

    fprintf('\nRoutes - movement time only:\n');
    for r = 1:numel(result.bestRoutes)
        fprintf('Route %d | travel time %.2f | jobs: ', r, result.bestRoutes(r).travelTime);
        disp(result.bestRoutes(r).routeOrder);
    end
end
