function result = runABCFlowShopVRPO(problem, params)
tic
% runABCFlowShopVRP_Optimized
%
% Optimized ABC algorithm for integrated permutation flow-shop +
% single-vehicle capacitated VRP problem.
%
% Objective:
%   Minimize:
%       deliveryFinish + idleWeight * totalMachineIdleTime
%
% where:
%   deliveryFinish = makespan + pure VRP movement time
%
% Main improvements:
%   1. Precomputed distance matrix
%   2. Cached route evaluations
%   3. NEH/SPT/LPT/geographical initial solutions
%   4. Faster flow-shop evaluation
%   5. Improved permutation neighborhoods
%   6. Elitism
%   7. Periodic local search
%   8. Adaptive neighborhood probabilities

    if nargin < 2 || isempty(params)
        params = struct();
    end

    problem = validateProblem(problem);
    nJobs = size(problem.P, 1);
    params = setDefaultParams(params, nJobs);

    if ~isempty(params.seed)
        rng(params.seed);
    else
        rng('shuffle');
    end

    % Precompute distance matrix once
    D = buildDistanceMatrix(problem.coords, problem.depot);

    % Cache for VRP route evaluations
    routeCache = containers.Map('KeyType', 'char', 'ValueType', 'any');

    % Initial heuristic base sequences
    baseSeqs = createBaseSequences(problem, D);

    % Initial move probabilities:
    % 1: swap, 2: insertion, 3: inversion, 4: block insertion, 5: 3-job rotation
    moveProb = params.initialMoveProb;
    moveScore = ones(1, 5);

    % Initial population
    sol = repmat(emptySolution(), params.SN, 1);

    for s = 1:params.SN
        seq = createInitialSequence(s, nJobs, params, baseSeqs);
        sol(s) = evaluateSequenceFast(seq, problem, params, D, routeCache, false);
    end

    [~, bestIdx] = min([sol.fitness]);
    best = sol(bestIdx);

    history = zeros(params.maxCycle, 1);

    % Main ABC loop
    for cycle = 1:params.maxCycle

        % ===============================================================
        % Employed Bee Phase
        % ===============================================================
        for s = 1:params.SN

            [candidateSeq, moveType] = makeNeighborABC(sol(s).sequence, moveProb);
            candidate = evaluateSequenceFast(candidateSeq, problem, params, D, routeCache, false);

            oldFitness = sol(s).fitness;
            [sol(s), improved] = greedySelection(sol(s), candidate);

            if improved
                improvement = max(0, oldFitness - candidate.fitness);
                moveScore(moveType) = moveScore(moveType) + 1 + improvement / max(1, abs(oldFitness));
            end
        end

        % ===============================================================
        % Onlooker Bee Phase
        % Rank-based probability prevents excessive domination
        % ===============================================================
        probs = rankSelectionProbabilities([sol.fitness]);

        for t = 1:params.SN

            s = rouletteWheelSelection(probs);

            [candidateSeq, moveType] = makeNeighborABC(sol(s).sequence, moveProb);
            candidate = evaluateSequenceFast(candidateSeq, problem, params, D, routeCache, false);

            oldFitness = sol(s).fitness;
            [sol(s), improved] = greedySelection(sol(s), candidate);

            if improved
                improvement = max(0, oldFitness - candidate.fitness);
                moveScore(moveType) = moveScore(moveType) + 1 + improvement / max(1, abs(oldFitness));
            end
        end

        % ===============================================================
        % Scout Bee Phase
        % Some scouts are generated around the global best solution
        % ===============================================================
        for s = 1:params.SN
            if sol(s).trial >= params.limit

                if rand < params.scoutBestRate
                    k = max(2, ceil(params.scoutPerturbRate * nJobs));
                    seq = perturbSequence(best.sequence, k);
                else
                    seq = createInitialSequence(s + cycle, nJobs, params, baseSeqs);
                end

                sol(s) = evaluateSequenceFast(seq, problem, params, D, routeCache, false);
            end
        end

        % ===============================================================
        % Update Best
        % ===============================================================
        [~, currentBestIdx] = min([sol.fitness]);

        if sol(currentBestIdx).fitness < best.fitness
            best = sol(currentBestIdx);
        end

        % ===============================================================
        % Periodic Local Search around best solution
        % ===============================================================
        if params.localSearchInterval > 0 && mod(cycle, params.localSearchInterval) == 0
            bestLS = localSearchBest(best, problem, params, D, routeCache, moveProb);

            if bestLS.fitness < best.fitness
                best = bestLS;
            end
        end

        % ===============================================================
        % Elitism: keep best solution inside population
        % ===============================================================
        [~, worstIdx] = max([sol.fitness]);

        if best.fitness < sol(worstIdx).fitness
            sol(worstIdx) = best;
            sol(worstIdx).trial = 0;
        end

        % ===============================================================
        % Adaptive neighborhood probability update
        % ===============================================================
        successProb = moveScore / sum(moveScore);
        moveProb = (1 - params.adaptRate) * moveProb + params.adaptRate * successProb;
        moveProb = moveProb / sum(moveProb);
        moveScore = ones(1, 5);

        history(cycle) = best.fitness;

        % if params.Display && mod(cycle, params.displayInterval) == 0
        %     fprintf('Cycle %4d | Best Fitness = %.6f | Makespan = %.4f | VRP = %.4f | Idle = %.4f\n', ...
        %         cycle, best.fitness, best.makespan, best.vrpTravelTime, best.idleTime);
        % end
    end

    % Re-evaluate best with full completion matrix and route details
    best = evaluateSequenceFast(best.sequence, problem, params, D, routeCache, true);

    result.bestSequence       = best.sequence;
    result.bestFitness        = best.fitness;
    result.bestMakespan       = best.makespan;
    result.bestIdleTime       = best.idleTime;
    result.bestIdleByMachine  = best.idleByMachine;
    result.bestVRPTravelTime  = best.vrpTravelTime;
    result.bestDeliveryFinish = best.deliveryFinish;
    result.bestRoutes         = best.routes;
    result.completionTimes    = best.completionTimes;
    result.history            = history;
    result.params             = params;
    result.finalMoveProb      = moveProb;
    result.routeCacheSize     = routeCache.Count;
    result.time               = toc;

end

% =======================================================================
% Parameter Settings
% =======================================================================
function params = setDefaultParams(params, nJobs)

    if ~isfield(params, 'SN') || isempty(params.SN)
        params.SN = 40;
    end

    if ~isfield(params, 'maxCycle') || isempty(params.maxCycle)
        params.maxCycle = 500;
    end

    if ~isfield(params, 'limit') || isempty(params.limit)
        params.limit = max(30, round(0.6 * nJobs));
    end

    if ~isfield(params, 'seed')
        params.seed = [];
    end

    if ~isfield(params, 'idleWeight') || isempty(params.idleWeight)
        params.idleWeight = 1;
    end

    if ~isfield(params, 'Display') || isempty(params.Display)
        params.Display = true;
    end

    if ~isfield(params, 'displayInterval') || isempty(params.displayInterval)
        params.displayInterval = 25;
    end

    % Initial solution parameters
    if ~isfield(params, 'initHeuristicRate') || isempty(params.initHeuristicRate)
        params.initHeuristicRate = 0.65;
    end

    if ~isfield(params, 'initPerturbRate') || isempty(params.initPerturbRate)
        params.initPerturbRate = 0.15;
    end

    % Scout phase
    if ~isfield(params, 'scoutBestRate') || isempty(params.scoutBestRate)
        params.scoutBestRate = 0.60;
    end

    if ~isfield(params, 'scoutPerturbRate') || isempty(params.scoutPerturbRate)
        params.scoutPerturbRate = 0.20;
    end

    % Local search
    if ~isfield(params, 'localSearchInterval') || isempty(params.localSearchInterval)
        params.localSearchInterval = 20;
    end

    if ~isfield(params, 'localSearchSamples') || isempty(params.localSearchSamples)
        params.localSearchSamples = max(30, 3 * nJobs);
    end

    % TSP parameters
    if ~isfield(params, 'exactTSPN') || isempty(params.exactTSPN)
        params.exactTSPN = 7;
    end

    if ~isfield(params, 'tspStarts') || isempty(params.tspStarts)
        params.tspStarts = 4;
    end

    if ~isfield(params, 'twoOptMaxPass') || isempty(params.twoOptMaxPass)
        params.twoOptMaxPass = 30;
    end

    % Adaptive neighborhood
    if ~isfield(params, 'adaptRate') || isempty(params.adaptRate)
        params.adaptRate = 0.15;
    end

    if ~isfield(params, 'initialMoveProb') || isempty(params.initialMoveProb)
        params.initialMoveProb = [0.25 0.25 0.25 0.15 0.10];
    end

    params.initialMoveProb = params.initialMoveProb / sum(params.initialMoveProb);

    % Idle time mode:
    % 'activeHorizon' keeps the logic close to your original code.
    % 'makespanHorizon' calculates idle time over [0, Cmax].
    if ~isfield(params, 'idleMode') || isempty(params.idleMode)
        params.idleMode = 'activeHorizon';
    end

    if params.SN < 1 || params.limit < 1 || params.maxCycle < 1
        error('params.SN, params.limit and params.maxCycle must be positive.');
    end

    if params.idleWeight < 0
        error('params.idleWeight must be nonnegative.');
    end
end

% =======================================================================
% Problem Validation
% =======================================================================
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

% =======================================================================
% Empty Solution
% =======================================================================
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

% =======================================================================
% Fast Evaluation
% =======================================================================
function sol = evaluateSequenceFast(seq, problem, params, D, routeCache, keepDetails)

    [makespan, C, idleTime, idleByMachine] = flowShopStats(seq, problem.P, keepDetails, params.idleMode);

    [vrpTravelTime, routes] = evaluateVRPMovementOnlyFast( ...
        seq, problem, params, D, routeCache, keepDetails);

    deliveryFinish = makespan + vrpTravelTime;

    fitness = deliveryFinish + params.idleWeight * idleTime;

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
    sol.trial           = 0;
end

% =======================================================================
% Faster Flowshop Evaluation
% =======================================================================
function [makespan, C, totalIdle, idleByMachine] = flowShopStats(seq, P, keepC, idleMode)

    n = numel(seq);
    m = size(P, 2);

    completion = zeros(1, m);

    if keepC
        C = zeros(n, m);
    else
        C = [];
    end

    firstJob = seq(1);

    if m == 1
        firstStart = 0;
    else
        firstStart = [0, cumsum(P(firstJob, 1:m-1))];
    end

    for i = 1:n
        job = seq(i);

        for machine = 1:m
            if machine == 1
                completion(machine) = completion(machine) + P(job, machine);
            else
                completion(machine) = max(completion(machine), completion(machine - 1)) + P(job, machine);
            end
        end

        if keepC
            C(i, :) = completion;
        end
    end

    makespan = completion(end);
    busyByMachine = sum(P, 1);

    switch lower(idleMode)
        case 'makespanhorizon'
            idleByMachine = max(0, makespan - busyByMachine);

        otherwise
            idleByMachine = max(0, completion - firstStart - busyByMachine);
    end

    totalIdle = sum(idleByMachine);
end

% =======================================================================
% VRP Evaluation with Route Cache
% =======================================================================
function [totalTravelTime, routes] = evaluateVRPMovementOnlyFast(seq, problem, params, D, routeCache, keepRoutes)

    batches = buildCapacityBatchesFast(seq, problem.weights, problem.capacity);
    nBatches = numel(batches);

    totalTravelTime = 0;

    if keepRoutes
        routes = repmat(struct( ...
            'jobs', [], ...
            'routeOrder', [], ...
            'travelTime', 0), nBatches, 1);
    else
        routes = [];
    end

    for b = 1:nBatches

        batchJobs = batches{b};

        [travelTime, routeOrder] = solveRouteCached(batchJobs, D, params, routeCache);

        totalTravelTime = totalTravelTime + travelTime;

        if keepRoutes
            routes(b).jobs       = batchJobs;
            routes(b).routeOrder = routeOrder;
            routes(b).travelTime = travelTime;
        end
    end
end

% =======================================================================
% Capacity Batching
% =======================================================================
function batches = buildCapacityBatchesFast(seq, weights, capacity)

    n = numel(seq);
    batches = cell(1, n);
    batchCount = 0;

    currentBatch = zeros(1, n);
    currentSize = 0;
    currentLoad = 0;

    for k = 1:n

        job = seq(k);
        w = weights(job);

        if currentLoad + w <= capacity

            currentSize = currentSize + 1;
            currentBatch(currentSize) = job;
            currentLoad = currentLoad + w;

        else

            batchCount = batchCount + 1;
            batches{batchCount} = currentBatch(1:currentSize);

            currentBatch(:) = 0;
            currentSize = 1;
            currentBatch(currentSize) = job;
            currentLoad = w;
        end
    end

    if currentSize > 0
        batchCount = batchCount + 1;
        batches{batchCount} = currentBatch(1:currentSize);
    end

    batches = batches(1:batchCount);
end

% =======================================================================
% Cached Route Solver
% =======================================================================
function [travelTime, routeOrder] = solveRouteCached(batchJobs, D, params, routeCache)

    key = routeKey(batchJobs);

    if isKey(routeCache, key)
        data = routeCache(key);
        travelTime = data.travelTime;
        routeOrder = data.routeOrder;
        return;
    end

    n = numel(batchJobs);

    if n == 0
        travelTime = 0;
        routeOrder = [];

    elseif n == 1
        routeOrder = batchJobs;
        travelTime = 2 * D(1, batchJobs(1) + 1);

    elseif n <= params.exactTSPN
        [travelTime, routeOrder] = exactRoute(batchJobs, D);

    else
        [travelTime, routeOrder] = heuristicRoute(batchJobs, D, params);
    end

    data.travelTime = travelTime;
    data.routeOrder = routeOrder;
    routeCache(key) = data;
end

function key = routeKey(jobs)

    jobs = sort(jobs);
    key = sprintf('%d_', jobs);
end

% =======================================================================
% Exact TSP for Small Routes
% =======================================================================
function [bestDistance, bestOrder] = exactRoute(batchJobs, D)

    allOrders = perms(batchJobs);

    bestDistance = inf;
    bestOrder = [];

    for r = 1:size(allOrders, 1)
        order = allOrders(r, :);
        d = routeDistanceActual(order, D);

        if d < bestDistance
            bestDistance = d;
            bestOrder = order;
        end
    end
end

% =======================================================================
% Heuristic TSP: Multi-start NN + 2-opt
% =======================================================================
function [bestDistance, bestOrder] = heuristicRoute(batchJobs, D, params)

    n = numel(batchJobs);

    depotDistances = D(1, batchJobs + 1);
    [~, idx] = sort(depotDistances, 'ascend');

    forcedStarts = batchJobs(idx(1:min(params.tspStarts, n)));

    bestDistance = inf;
    bestOrder = [];

    % Natural nearest neighbor
    order = nearestNeighborOrder(batchJobs, D, []);
    [order, d] = twoOptImproveFast(order, D, params.twoOptMaxPass);

    if d < bestDistance
        bestDistance = d;
        bestOrder = order;
    end

    % Forced-start nearest neighbors
    for s = 1:numel(forcedStarts)

        order = nearestNeighborOrder(batchJobs, D, forcedStarts(s));
        [order, d] = twoOptImproveFast(order, D, params.twoOptMaxPass);

        if d < bestDistance
            bestDistance = d;
            bestOrder = order;
        end
    end
end

function order = nearestNeighborOrder(batchJobs, D, firstJob)

    remaining = batchJobs(:)';
    order = [];

    if isempty(firstJob)
        currentNode = 1;
    else
        order = firstJob;
        remaining(remaining == firstJob) = [];
        currentNode = firstJob + 1;
    end

    while ~isempty(remaining)

        distances = D(currentNode, remaining + 1);
        [~, idx] = min(distances);

        nextJob = remaining(idx);
        order = [order, nextJob]; %#ok<AGROW>

        remaining(idx) = [];
        currentNode = nextJob + 1;
    end
end

function [bestOrder, bestDistance] = twoOptImproveFast(order, D, maxPass)

    bestOrder = order;
    n = numel(bestOrder);
    bestDistance = routeDistanceActual(bestOrder, D);

    if n < 3
        return;
    end

    improved = true;
    pass = 0;

    while improved && pass < maxPass

        improved = false;
        pass = pass + 1;

        for i = 1:n-1
            for j = i+1:n

                if i == 1
                    A = 1;
                else
                    A = bestOrder(i-1) + 1;
                end

                B = bestOrder(i) + 1;
                C = bestOrder(j) + 1;

                if j == n
                    E = 1;
                else
                    E = bestOrder(j+1) + 1;
                end

                oldEdge = D(A, B) + D(C, E);
                newEdge = D(A, C) + D(B, E);

                delta = newEdge - oldEdge;

                if delta < -1e-12
                    bestOrder(i:j) = bestOrder(j:-1:i);
                    bestDistance = bestDistance + delta;
                    improved = true;
                end
            end
        end
    end
end

function d = routeDistanceActual(order, D)

    if isempty(order)
        d = 0;
        return;
    end

    d = D(1, order(1) + 1);

    for k = 1:numel(order)-1
        d = d + D(order(k) + 1, order(k+1) + 1);
    end

    d = d + D(order(end) + 1, 1);
end

% =======================================================================
% Distance Matrix
% Node 1       : depot
% Node job + 1 : customer/job
% =======================================================================
function D = buildDistanceMatrix(coords, depot)

    points = [depot; coords];
    n = size(points, 1);

    D = zeros(n, n);

    for i = 1:n
        D(i, :) = sum(abs(points(i, :) - points), 2)';
    end
end

% =======================================================================
% Initial Population
% =======================================================================
function baseSeqs = createBaseSequences(problem, D)

    P = problem.P;
    nJobs = size(P, 1);

    baseSeqs = cell(1, 4);

    baseSeqs{1} = nehSequence(P);
    baseSeqs{2} = geoNearestSequence(D, nJobs);

    totalProcessing = sum(P, 2);

    [~, spt] = sort(totalProcessing, 'ascend');
    [~, lpt] = sort(totalProcessing, 'descend');

    baseSeqs{3} = spt(:)';
    baseSeqs{4} = lpt(:)';
end

function seq = createInitialSequence(s, nJobs, params, baseSeqs)

    nBase = numel(baseSeqs);

    if s <= nBase
        seq = baseSeqs{s};
        return;
    end

    if rand < params.initHeuristicRate
        base = baseSeqs{randi(nBase)};
        k = max(1, ceil(params.initPerturbRate * nJobs));
        seq = perturbSequence(base, randi(k));
    else
        seq = randperm(nJobs);
    end
end

function seq = nehSequence(P)

    nJobs = size(P, 1);
    totalProcessing = sum(P, 2);

    [~, order] = sort(totalProcessing, 'descend');

    seq = [];

    for k = 1:nJobs

        job = order(k);
        bestSeq = [];
        bestCmax = inf;

        for pos = 1:k

            candidate = [seq(1:pos-1), job, seq(pos:end)];

            cmax = flowShopStats(candidate, P, false, 'activeHorizon');

            if cmax < bestCmax
                bestCmax = cmax;
                bestSeq = candidate;
            end
        end

        seq = bestSeq;
    end
end

function seq = geoNearestSequence(D, nJobs)

    remaining = 1:nJobs;
    seq = zeros(1, nJobs);
    currentNode = 1;

    for k = 1:nJobs

        distances = D(currentNode, remaining + 1);
        [~, idx] = min(distances);

        nextJob = remaining(idx);
        seq(k) = nextJob;

        remaining(idx) = [];
        currentNode = nextJob + 1;
    end
end

function seq = perturbSequence(seq, k)

    for t = 1:k
        [seq, ~] = makeNeighborABC(seq, [0.30 0.30 0.25 0.10 0.05]);
    end
end

% =======================================================================
% ABC Neighborhoods
% =======================================================================
function [neigh, moveType] = makeNeighborABC(seq, moveProb)

    n = numel(seq);
    neigh = seq;

    if n < 2
        moveType = 1;
        return;
    end

    moveType = rouletteWheelSelection(moveProb);

    switch moveType

        case 1
            % Swap
            idx = randperm(n, 2);
            neigh(idx) = neigh(fliplr(idx));

        case 2
            % Insertion
            idx = randperm(n, 2);
            i = idx(1);
            j = idx(2);

            job = neigh(i);
            neigh(i) = [];

            if j > numel(neigh) + 1
                j = numel(neigh) + 1;
            end

            neigh = [neigh(1:j-1), job, neigh(j:end)];

        case 3
            % Inversion
            idx = sort(randperm(n, 2));
            i = idx(1);
            j = idx(2);

            neigh(i:j) = neigh(j:-1:i);

        case 4
            % Block insertion
            if n < 4
                idx = randperm(n, 2);
                neigh(idx) = neigh(fliplr(idx));
            else
                idx = sort(randperm(n, 2));
                i = idx(1);
                j = idx(2);

                block = neigh(i:j);
                rest = [neigh(1:i-1), neigh(j+1:end)];

                pos = randi(numel(rest) + 1);
                neigh = [rest(1:pos-1), block, rest(pos:end)];
            end

        case 5
            % 3-job rotation
            if n < 3
                idx = randperm(n, 2);
                neigh(idx) = neigh(fliplr(idx));
            else
                idx = sort(randperm(n, 3));
                temp = neigh(idx);
                neigh(idx) = temp([2 3 1]);
            end
    end
end

% =======================================================================
% Selection
% =======================================================================
function [selected, improved] = greedySelection(current, candidate)

    if candidate.fitness < current.fitness
        selected = candidate;
        selected.trial = 0;
        improved = true;
    else
        selected = current;
        selected.trial = current.trial + 1;
        improved = false;
    end
end

function probs = rankSelectionProbabilities(fitnessValues)

    n = numel(fitnessValues);
    [~, order] = sort(fitnessValues, 'ascend');

    score = zeros(1, n);

    for r = 1:n
        score(order(r)) = n - r + 1;
    end

    probs = score / sum(score);
end

function index = rouletteWheelSelection(prob)

    prob = prob(:)' / sum(prob);
    c = cumsum(prob);
    r = rand;

    index = find(r <= c, 1, 'first');

    if isempty(index)
        index = numel(prob);
    end
end

% =======================================================================
% Local Search around Best
% =======================================================================
function bestLS = localSearchBest(best, problem, params, D, routeCache, moveProb)

    bestLS = best;

    for q = 1:params.localSearchSamples

        [candidateSeq, ~] = makeNeighborABC(bestLS.sequence, moveProb);

        candidate = evaluateSequenceFast(candidateSeq, problem, params, D, routeCache, false);

        if candidate.fitness < bestLS.fitness
            bestLS = candidate;
        end
    end
end