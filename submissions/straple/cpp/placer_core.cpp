#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

constexpr double kOverlapGap = 0.05;

struct PlacerState
{
    int numHardMacros = 0;
    double canvasWidth = 0.0;
    double canvasHeight = 0.0;

    std::vector<double> sizeX;
    std::vector<double> sizeY;
    std::vector<double> halfWidth;
    std::vector<double> halfHeight;
    std::vector<char> movable;

    std::vector<double> posX;
    std::vector<double> posY;

    std::vector<int32_t> edgeA;
    std::vector<int32_t> edgeB;
    std::vector<double> edgeWeight;

    std::vector<std::vector<int32_t>> adjacencyIndex;
    std::vector<std::vector<double>> adjacencyWeight;

    std::mt19937_64 rng;
};

inline double clampDouble(double value, double low, double high)
{
    if (value < low)
    {
        return low;
    }
    if (value > high)
    {
        return high;
    }
    return value;
}

bool checkSingleOverlap(const PlacerState& state, int32_t macroIndex)
{
    const double xi = state.posX[macroIndex];
    const double yi = state.posY[macroIndex];
    const double wi = state.sizeX[macroIndex];
    const double hi = state.sizeY[macroIndex];
    const double gap = kOverlapGap;
    const int totalMacros = state.numHardMacros;
    for (int j = 0; j < totalMacros; ++j)
    {
        if (j == macroIndex)
        {
            continue;
        }
        const double sepX = (wi + state.sizeX[j]) * 0.5 + gap;
        const double sepY = (hi + state.sizeY[j]) * 0.5 + gap;
        if (std::fabs(xi - state.posX[j]) < sepX && std::fabs(yi - state.posY[j]) < sepY)
        {
            return true;
        }
    }
    return false;
}

double computeWirelength(const PlacerState& state)
{
    const std::size_t numEdges = state.edgeA.size();
    double total = 0.0;
    for (std::size_t k = 0; k < numEdges; ++k)
    {
        const int32_t a = state.edgeA[k];
        const int32_t b = state.edgeB[k];
        const double dx = std::fabs(state.posX[a] - state.posX[b]);
        const double dy = std::fabs(state.posY[a] - state.posY[b]);
        total += state.edgeWeight[k] * (dx + dy);
    }
    return total;
}

void initializePlacerState(
    PlacerState& state,
    py::array_t<double> initialPos,
    py::array_t<double> sizes,
    py::array_t<bool> movableMask,
    py::array_t<int32_t> edges,
    py::array_t<double> edgeWeights,
    double canvasWidth,
    double canvasHeight,
    uint64_t seed)
{
    auto posBuffer = initialPos.unchecked<2>();
    auto sizeBuffer = sizes.unchecked<2>();
    auto movableBuffer = movableMask.unchecked<1>();

    const int n = static_cast<int>(posBuffer.shape(0));
    state.numHardMacros = n;
    state.canvasWidth = canvasWidth;
    state.canvasHeight = canvasHeight;

    state.posX.assign(n, 0.0);
    state.posY.assign(n, 0.0);
    state.sizeX.assign(n, 0.0);
    state.sizeY.assign(n, 0.0);
    state.halfWidth.assign(n, 0.0);
    state.halfHeight.assign(n, 0.0);
    state.movable.assign(n, 0);

    for (int i = 0; i < n; ++i)
    {
        state.posX[i] = posBuffer(i, 0);
        state.posY[i] = posBuffer(i, 1);
        state.sizeX[i] = sizeBuffer(i, 0);
        state.sizeY[i] = sizeBuffer(i, 1);
        state.halfWidth[i] = state.sizeX[i] * 0.5;
        state.halfHeight[i] = state.sizeY[i] * 0.5;
        state.movable[i] = movableBuffer(i) ? 1 : 0;
    }

    auto edgeBuffer = edges.unchecked<2>();
    auto weightBuffer = edgeWeights.unchecked<1>();
    const int numEdges = static_cast<int>(edgeBuffer.shape(0));
    state.edgeA.resize(numEdges);
    state.edgeB.resize(numEdges);
    state.edgeWeight.resize(numEdges);
    for (int e = 0; e < numEdges; ++e)
    {
        state.edgeA[e] = edgeBuffer(e, 0);
        state.edgeB[e] = edgeBuffer(e, 1);
        state.edgeWeight[e] = weightBuffer(e);
    }

    state.adjacencyIndex.assign(n, {});
    state.adjacencyWeight.assign(n, {});
    for (int e = 0; e < numEdges; ++e)
    {
        const int32_t a = state.edgeA[e];
        const int32_t b = state.edgeB[e];
        const double w = state.edgeWeight[e];
        state.adjacencyIndex[a].push_back(b);
        state.adjacencyIndex[b].push_back(a);
        state.adjacencyWeight[a].push_back(w);
        state.adjacencyWeight[b].push_back(w);
    }

    state.rng.seed(seed);
}

void legalize(PlacerState& state)
{
    const int n = state.numHardMacros;

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i)
    {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&state](int a, int b) {
        return state.sizeX[a] * state.sizeY[a] > state.sizeX[b] * state.sizeY[b];
    });

    std::vector<char> placed(n, 0);
    std::vector<double> initialX = state.posX;
    std::vector<double> initialY = state.posY;

    auto hasConflict = [&](int idx, double cx, double cy) {
        for (int j = 0; j < n; ++j)
        {
            if (!placed[j] || j == idx)
            {
                continue;
            }
            const double sepX = (state.sizeX[idx] + state.sizeX[j]) * 0.5 + kOverlapGap;
            const double sepY = (state.sizeY[idx] + state.sizeY[j]) * 0.5 + kOverlapGap;
            if (std::fabs(cx - state.posX[j]) < sepX && std::fabs(cy - state.posY[j]) < sepY)
            {
                return true;
            }
        }
        return false;
    };

    for (int idx : order)
    {
        if (!state.movable[idx])
        {
            placed[idx] = 1;
            continue;
        }

        if (!hasConflict(idx, state.posX[idx], state.posY[idx]))
        {
            placed[idx] = 1;
            continue;
        }

        const double step = std::max(state.sizeX[idx], state.sizeY[idx]) * 0.25;
        double bestX = state.posX[idx];
        double bestY = state.posY[idx];
        double bestDistance = std::numeric_limits<double>::infinity();
        bool found = false;

        for (int r = 1; r < 150 && !found; ++r)
        {
            for (int dxm = -r; dxm <= r; ++dxm)
            {
                for (int dym = -r; dym <= r; ++dym)
                {
                    if (std::abs(dxm) != r && std::abs(dym) != r)
                    {
                        continue;
                    }
                    const double cx = clampDouble(
                        initialX[idx] + dxm * step,
                        state.halfWidth[idx],
                        state.canvasWidth - state.halfWidth[idx]);
                    const double cy = clampDouble(
                        initialY[idx] + dym * step,
                        state.halfHeight[idx],
                        state.canvasHeight - state.halfHeight[idx]);
                    if (hasConflict(idx, cx, cy))
                    {
                        continue;
                    }
                    const double distance =
                        (cx - initialX[idx]) * (cx - initialX[idx]) +
                        (cy - initialY[idx]) * (cy - initialY[idx]);
                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        bestX = cx;
                        bestY = cy;
                        found = true;
                    }
                }
            }
        }

        state.posX[idx] = bestX;
        state.posY[idx] = bestY;
        placed[idx] = 1;
    }
}

int legalizeMinDisplacement(PlacerState& state, int maxIters)
{
    const int n = state.numHardMacros;
    int totalMoves = 0;

    for (int iter = 0; iter < maxIters; ++iter)
    {
        bool anyConflict = false;
        for (int i = 0; i < n; ++i)
        {
            if (!state.movable[i])
            {
                continue;
            }
            const double xi = state.posX[i];
            const double yi = state.posY[i];
            for (int j = 0; j < n; ++j)
            {
                if (j == i)
                {
                    continue;
                }
                const double sepX = (state.sizeX[i] + state.sizeX[j]) * 0.5 + kOverlapGap;
                const double sepY = (state.sizeY[i] + state.sizeY[j]) * 0.5 + kOverlapGap;
                const double dx = xi - state.posX[j];
                const double dy = yi - state.posY[j];
                const double absdx = std::fabs(dx);
                const double absdy = std::fabs(dy);
                if (absdx < sepX && absdy < sepY)
                {
                    anyConflict = true;
                    const double overlapX = sepX - absdx;
                    const double overlapY = sepY - absdy;
                    if (overlapX < overlapY)
                    {
                        const double sign = (dx >= 0.0) ? 1.0 : -1.0;
                        const double push = overlapX * 0.5 + 1e-6;
                        if (state.movable[j])
                        {
                            state.posX[i] = clampDouble(xi + sign * push,
                                state.halfWidth[i], state.canvasWidth - state.halfWidth[i]);
                            state.posX[j] = clampDouble(state.posX[j] - sign * push,
                                state.halfWidth[j], state.canvasWidth - state.halfWidth[j]);
                        }
                        else
                        {
                            state.posX[i] = clampDouble(xi + sign * (overlapX + 1e-6),
                                state.halfWidth[i], state.canvasWidth - state.halfWidth[i]);
                        }
                    }
                    else
                    {
                        const double sign = (dy >= 0.0) ? 1.0 : -1.0;
                        const double push = overlapY * 0.5 + 1e-6;
                        if (state.movable[j])
                        {
                            state.posY[i] = clampDouble(yi + sign * push,
                                state.halfHeight[i], state.canvasHeight - state.halfHeight[i]);
                            state.posY[j] = clampDouble(state.posY[j] - sign * push,
                                state.halfHeight[j], state.canvasHeight - state.halfHeight[j]);
                        }
                        else
                        {
                            state.posY[i] = clampDouble(yi + sign * (overlapY + 1e-6),
                                state.halfHeight[i], state.canvasHeight - state.halfHeight[i]);
                        }
                    }
                    ++totalMoves;
                    break;
                }
            }
        }
        if (!anyConflict)
        {
            break;
        }
    }
    return totalMoves;
}

struct SAStats
{
    int numIters = 0;
    int numAccepted = 0;
    int numRejected = 0;
    int numAcceptedBoltzmann = 0;
    int numRejectedOverlap = 0;
    int numSwap = 0;
    int numShift = 0;
    int numTowardNeighbor = 0;
    int bestStep = 0;
    double initialCost = 0.0;
    double bestCost = 0.0;
    double finalCost = 0.0;
    std::vector<double> trajectoryWL;
    std::vector<int> trajectorySteps;
};

py::dict simulatedAnnealingRefineWithStats(PlacerState& state, int numIters, int snapshotEvery);

void simulatedAnnealingRefine(PlacerState& state, int numIters)
{
    if (numIters <= 0 || state.edgeA.empty())
    {
        return;
    }

    std::vector<int> movableIndices;
    movableIndices.reserve(state.numHardMacros);
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        if (state.movable[i])
        {
            movableIndices.push_back(i);
        }
    }
    if (movableIndices.empty())
    {
        return;
    }

    std::vector<double> bestPosX = state.posX;
    std::vector<double> bestPosY = state.posY;
    double currentCost = computeWirelength(state);
    double bestCost = currentCost;

    const double maxCanvas = std::max(state.canvasWidth, state.canvasHeight);
    const double tStart = maxCanvas * 0.15;
    const double tEnd = maxCanvas * 0.001;

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> standardNormal(0.0, 1.0);

    for (int step = 0; step < numIters; ++step)
    {
        const double frac = static_cast<double>(step) / numIters;
        const double temperature = tStart * std::pow(tEnd / tStart, frac);

        const int chosen = movableIndices[std::uniform_int_distribution<int>(
            0, static_cast<int>(movableIndices.size()) - 1)(state.rng)];
        const double oldX = state.posX[chosen];
        const double oldY = state.posY[chosen];

        const double moveType = uniform01(state.rng);
        bool isSwap = false;
        int swapPartner = -1;
        double oldPartnerX = 0.0;
        double oldPartnerY = 0.0;

        if (moveType < 0.5)
        {
            const double shift = temperature * (0.3 + 0.7 * (1.0 - frac));
            state.posX[chosen] = clampDouble(
                oldX + standardNormal(state.rng) * shift,
                state.halfWidth[chosen],
                state.canvasWidth - state.halfWidth[chosen]);
            state.posY[chosen] = clampDouble(
                oldY + standardNormal(state.rng) * shift,
                state.halfHeight[chosen],
                state.canvasHeight - state.halfHeight[chosen]);
        }
        else if (moveType < 0.8)
        {
            const auto& neighborList = state.adjacencyIndex[chosen];
            int partner = -1;
            if (!neighborList.empty() && uniform01(state.rng) < 0.7)
            {
                std::vector<int> movableNeighbors;
                movableNeighbors.reserve(neighborList.size());
                for (int n : neighborList)
                {
                    if (state.movable[n])
                    {
                        movableNeighbors.push_back(n);
                    }
                }
                if (!movableNeighbors.empty())
                {
                    partner = movableNeighbors[std::uniform_int_distribution<int>(
                        0, static_cast<int>(movableNeighbors.size()) - 1)(state.rng)];
                }
            }
            if (partner < 0)
            {
                partner = movableIndices[std::uniform_int_distribution<int>(
                    0, static_cast<int>(movableIndices.size()) - 1)(state.rng)];
            }
            if (partner == chosen)
            {
                continue;
            }
            isSwap = true;
            swapPartner = partner;
            oldPartnerX = state.posX[partner];
            oldPartnerY = state.posY[partner];
            state.posX[chosen] = clampDouble(
                oldPartnerX,
                state.halfWidth[chosen],
                state.canvasWidth - state.halfWidth[chosen]);
            state.posY[chosen] = clampDouble(
                oldPartnerY,
                state.halfHeight[chosen],
                state.canvasHeight - state.halfHeight[chosen]);
            state.posX[partner] = clampDouble(
                oldX,
                state.halfWidth[partner],
                state.canvasWidth - state.halfWidth[partner]);
            state.posY[partner] = clampDouble(
                oldY,
                state.halfHeight[partner],
                state.canvasHeight - state.halfHeight[partner]);
            if (checkSingleOverlap(state, chosen) || checkSingleOverlap(state, partner))
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
                state.posX[partner] = oldPartnerX;
                state.posY[partner] = oldPartnerY;
                continue;
            }
        }
        else
        {
            const auto& neighborList = state.adjacencyIndex[chosen];
            if (!neighborList.empty())
            {
                const int partner = neighborList[std::uniform_int_distribution<std::size_t>(
                    0, neighborList.size() - 1)(state.rng)];
                const double alpha = 0.05 + 0.25 * uniform01(state.rng);
                state.posX[chosen] = clampDouble(
                    oldX + alpha * (state.posX[partner] - oldX),
                    state.halfWidth[chosen],
                    state.canvasWidth - state.halfWidth[chosen]);
                state.posY[chosen] = clampDouble(
                    oldY + alpha * (state.posY[partner] - oldY),
                    state.halfHeight[chosen],
                    state.canvasHeight - state.halfHeight[chosen]);
            }
        }

        if (!isSwap)
        {
            if (checkSingleOverlap(state, chosen))
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
                continue;
            }
        }

        const double newCost = computeWirelength(state);
        const double delta = newCost - currentCost;
        bool accept = false;
        if (delta < 0.0)
        {
            accept = true;
        }
        else
        {
            const double probability = std::exp(-delta / std::max(temperature, 1e-12));
            accept = uniform01(state.rng) < probability;
        }
        if (accept)
        {
            currentCost = newCost;
            if (currentCost < bestCost)
            {
                bestCost = currentCost;
                bestPosX = state.posX;
                bestPosY = state.posY;
            }
        }
        else
        {
            if (isSwap)
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
                state.posX[swapPartner] = oldPartnerX;
                state.posY[swapPartner] = oldPartnerY;
            }
            else
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
            }
        }
    }

    state.posX = bestPosX;
    state.posY = bestPosY;
}

py::dict simulatedAnnealingRefineWithStats(PlacerState& state, int numIters, int snapshotEvery)
{
    py::dict result;
    if (numIters <= 0 || state.edgeA.empty())
    {
        result["skipped"] = true;
        result["reason"] = state.edgeA.empty() ? "no_edges" : "no_iters";
        return result;
    }

    std::vector<int> movableIndices;
    movableIndices.reserve(state.numHardMacros);
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        if (state.movable[i])
        {
            movableIndices.push_back(i);
        }
    }
    if (movableIndices.empty())
    {
        result["skipped"] = true;
        result["reason"] = "no_movable";
        return result;
    }

    std::vector<double> bestPosX = state.posX;
    std::vector<double> bestPosY = state.posY;
    double currentCost = computeWirelength(state);
    double bestCost = currentCost;
    const double initialCost = currentCost;

    const double maxCanvas = std::max(state.canvasWidth, state.canvasHeight);
    const double tStart = maxCanvas * 0.15;
    const double tEnd = maxCanvas * 0.001;

    std::uniform_real_distribution<double> uniform01(0.0, 1.0);
    std::normal_distribution<double> standardNormal(0.0, 1.0);

    int numAccepted = 0;
    int numRejected = 0;
    int numAcceptedBoltzmann = 0;
    int numRejectedOverlap = 0;
    int numSwap = 0;
    int numShift = 0;
    int numTowardNeighbor = 0;
    int bestStep = 0;
    std::vector<double> trajWL;
    std::vector<int> trajSteps;
    std::vector<double> trajTemp;

    for (int step = 0; step < numIters; ++step)
    {
        const double frac = static_cast<double>(step) / numIters;
        const double temperature = tStart * std::pow(tEnd / tStart, frac);

        const int chosen = movableIndices[std::uniform_int_distribution<int>(
            0, static_cast<int>(movableIndices.size()) - 1)(state.rng)];
        const double oldX = state.posX[chosen];
        const double oldY = state.posY[chosen];

        const double moveType = uniform01(state.rng);
        bool isSwap = false;
        int swapPartner = -1;
        double oldPartnerX = 0.0;
        double oldPartnerY = 0.0;

        if (moveType < 0.5)
        {
            ++numShift;
            const double shift = temperature * (0.3 + 0.7 * (1.0 - frac));
            state.posX[chosen] = clampDouble(
                oldX + standardNormal(state.rng) * shift,
                state.halfWidth[chosen],
                state.canvasWidth - state.halfWidth[chosen]);
            state.posY[chosen] = clampDouble(
                oldY + standardNormal(state.rng) * shift,
                state.halfHeight[chosen],
                state.canvasHeight - state.halfHeight[chosen]);
        }
        else if (moveType < 0.8)
        {
            ++numSwap;
            const auto& neighborList = state.adjacencyIndex[chosen];
            int partner = -1;
            if (!neighborList.empty() && uniform01(state.rng) < 0.7)
            {
                std::vector<int> movableNeighbors;
                movableNeighbors.reserve(neighborList.size());
                for (int n : neighborList)
                {
                    if (state.movable[n])
                    {
                        movableNeighbors.push_back(n);
                    }
                }
                if (!movableNeighbors.empty())
                {
                    partner = movableNeighbors[std::uniform_int_distribution<int>(
                        0, static_cast<int>(movableNeighbors.size()) - 1)(state.rng)];
                }
            }
            if (partner < 0)
            {
                partner = movableIndices[std::uniform_int_distribution<int>(
                    0, static_cast<int>(movableIndices.size()) - 1)(state.rng)];
            }
            if (partner == chosen)
            {
                continue;
            }
            isSwap = true;
            swapPartner = partner;
            oldPartnerX = state.posX[partner];
            oldPartnerY = state.posY[partner];
            state.posX[chosen] = clampDouble(
                oldPartnerX,
                state.halfWidth[chosen],
                state.canvasWidth - state.halfWidth[chosen]);
            state.posY[chosen] = clampDouble(
                oldPartnerY,
                state.halfHeight[chosen],
                state.canvasHeight - state.halfHeight[chosen]);
            state.posX[partner] = clampDouble(
                oldX,
                state.halfWidth[partner],
                state.canvasWidth - state.halfWidth[partner]);
            state.posY[partner] = clampDouble(
                oldY,
                state.halfHeight[partner],
                state.canvasHeight - state.halfHeight[partner]);
            if (checkSingleOverlap(state, chosen) || checkSingleOverlap(state, partner))
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
                state.posX[partner] = oldPartnerX;
                state.posY[partner] = oldPartnerY;
                ++numRejectedOverlap;
                ++numRejected;
                continue;
            }
        }
        else
        {
            ++numTowardNeighbor;
            const auto& neighborList = state.adjacencyIndex[chosen];
            if (!neighborList.empty())
            {
                int partner = neighborList[std::uniform_int_distribution<int>(
                    0, static_cast<int>(neighborList.size()) - 1)(state.rng)];
                const double alpha = 0.1 + 0.4 * uniform01(state.rng);
                state.posX[chosen] = clampDouble(
                    oldX + alpha * (state.posX[partner] - oldX),
                    state.halfWidth[chosen],
                    state.canvasWidth - state.halfWidth[chosen]);
                state.posY[chosen] = clampDouble(
                    oldY + alpha * (state.posY[partner] - oldY),
                    state.halfHeight[chosen],
                    state.canvasHeight - state.halfHeight[chosen]);
            }
        }

        if (!isSwap)
        {
            if (checkSingleOverlap(state, chosen))
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
                ++numRejectedOverlap;
                ++numRejected;
                continue;
            }
        }

        const double newCost = computeWirelength(state);
        const double delta = newCost - currentCost;
        bool accept = false;
        bool boltzmann = false;
        if (delta < 0.0)
        {
            accept = true;
        }
        else
        {
            const double probability = std::exp(-delta / std::max(temperature, 1e-12));
            accept = uniform01(state.rng) < probability;
            if (accept)
            {
                boltzmann = true;
            }
        }
        if (accept)
        {
            ++numAccepted;
            if (boltzmann)
            {
                ++numAcceptedBoltzmann;
            }
            currentCost = newCost;
            if (currentCost < bestCost)
            {
                bestCost = currentCost;
                bestPosX = state.posX;
                bestPosY = state.posY;
                bestStep = step;
            }
        }
        else
        {
            ++numRejected;
            if (isSwap)
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
                state.posX[swapPartner] = oldPartnerX;
                state.posY[swapPartner] = oldPartnerY;
            }
            else
            {
                state.posX[chosen] = oldX;
                state.posY[chosen] = oldY;
            }
        }

        if (snapshotEvery > 0 && (step % snapshotEvery == 0 || step == numIters - 1))
        {
            trajSteps.push_back(step);
            trajWL.push_back(currentCost);
            trajTemp.push_back(temperature);
        }
    }

    state.posX = bestPosX;
    state.posY = bestPosY;

    result["skipped"] = false;
    result["num_iters"] = numIters;
    result["num_accepted"] = numAccepted;
    result["num_rejected"] = numRejected;
    result["num_accepted_boltzmann"] = numAcceptedBoltzmann;
    result["num_rejected_overlap"] = numRejectedOverlap;
    result["num_shift"] = numShift;
    result["num_swap"] = numSwap;
    result["num_toward_neighbor"] = numTowardNeighbor;
    result["best_step"] = bestStep;
    result["initial_wl"] = initialCost;
    result["best_wl"] = bestCost;
    result["final_wl"] = currentCost;
    result["t_start"] = tStart;
    result["t_end"] = tEnd;
    if (snapshotEvery > 0)
    {
        result["trajectory_steps"] = trajSteps;
        result["trajectory_wl"] = trajWL;
        result["trajectory_temp"] = trajTemp;
    }
    return result;
}

bool repairMacro(PlacerState& state, int macroIndex)
{
    const auto& neighborList = state.adjacencyIndex[macroIndex];
    const auto& neighborWeights = state.adjacencyWeight[macroIndex];

    double centroidX = state.posX[macroIndex];
    double centroidY = state.posY[macroIndex];

    if (!neighborList.empty())
    {
        double totalWeight = 0.0;
        double sumX = 0.0;
        double sumY = 0.0;
        for (std::size_t k = 0; k < neighborList.size(); ++k)
        {
            const int j = neighborList[k];
            const double w = neighborWeights[k];
            sumX += state.posX[j] * w;
            sumY += state.posY[j] * w;
            totalWeight += w;
        }
        if (totalWeight > 0.0)
        {
            centroidX = sumX / totalWeight;
            centroidY = sumY / totalWeight;
        }
    }

    centroidX = clampDouble(centroidX, state.halfWidth[macroIndex], state.canvasWidth - state.halfWidth[macroIndex]);
    centroidY = clampDouble(centroidY, state.halfHeight[macroIndex], state.canvasHeight - state.halfHeight[macroIndex]);

    const double oldX = state.posX[macroIndex];
    const double oldY = state.posY[macroIndex];

    state.posX[macroIndex] = centroidX;
    state.posY[macroIndex] = centroidY;
    if (!checkSingleOverlap(state, macroIndex))
    {
        return true;
    }

    const double step = std::max(state.sizeX[macroIndex], state.sizeY[macroIndex]) * 0.25;
    for (int r = 1; r < 80; ++r)
    {
        for (int dxm = -r; dxm <= r; ++dxm)
        {
            for (int dym = -r; dym <= r; ++dym)
            {
                if (std::abs(dxm) != r && std::abs(dym) != r)
                {
                    continue;
                }
                const double trialX = clampDouble(
                    centroidX + dxm * step,
                    state.halfWidth[macroIndex],
                    state.canvasWidth - state.halfWidth[macroIndex]);
                const double trialY = clampDouble(
                    centroidY + dym * step,
                    state.halfHeight[macroIndex],
                    state.canvasHeight - state.halfHeight[macroIndex]);
                state.posX[macroIndex] = trialX;
                state.posY[macroIndex] = trialY;
                if (!checkSingleOverlap(state, macroIndex))
                {
                    return true;
                }
            }
        }
    }
    state.posX[macroIndex] = oldX;
    state.posY[macroIndex] = oldY;
    return false;
}

double computeCongScoreForBox(
    double xMin, double xMax, double yMin, double yMax,
    const double* congGrid, int gridRows, int gridCols,
    double canvasWidth, double canvasHeight)
{
    const double cellW = canvasWidth / gridCols;
    const double cellH = canvasHeight / gridRows;
    int colMin = static_cast<int>(std::floor(xMin / cellW));
    int colMax = static_cast<int>(std::floor(xMax / cellW));
    int rowMin = static_cast<int>(std::floor(yMin / cellH));
    int rowMax = static_cast<int>(std::floor(yMax / cellH));
    if (colMin < 0)
    {
        colMin = 0;
    }
    if (colMax >= gridCols)
    {
        colMax = gridCols - 1;
    }
    if (rowMin < 0)
    {
        rowMin = 0;
    }
    if (rowMax >= gridRows)
    {
        rowMax = gridRows - 1;
    }
    if (colMin > colMax || rowMin > rowMax)
    {
        return 0.0;
    }
    double maxCong = 0.0;
    for (int row = rowMin; row <= rowMax; ++row)
    {
        for (int col = colMin; col <= colMax; ++col)
        {
            const double value = congGrid[row * gridCols + col];
            if (value > maxCong)
            {
                maxCong = value;
            }
        }
    }
    return maxCong;
}

bool repairMacroAware(
    PlacerState& state, int macroIndex,
    const double* congGrid, int gridRows, int gridCols,
    double threshold)
{
    const auto& neighborList = state.adjacencyIndex[macroIndex];
    const auto& neighborWeights = state.adjacencyWeight[macroIndex];

    double centroidX = state.posX[macroIndex];
    double centroidY = state.posY[macroIndex];

    if (!neighborList.empty())
    {
        double totalWeight = 0.0;
        double sumX = 0.0;
        double sumY = 0.0;
        for (std::size_t k = 0; k < neighborList.size(); ++k)
        {
            const int j = neighborList[k];
            const double w = neighborWeights[k];
            sumX += state.posX[j] * w;
            sumY += state.posY[j] * w;
            totalWeight += w;
        }
        if (totalWeight > 0.0)
        {
            centroidX = sumX / totalWeight;
            centroidY = sumY / totalWeight;
        }
    }

    centroidX = clampDouble(centroidX, state.halfWidth[macroIndex], state.canvasWidth - state.halfWidth[macroIndex]);
    centroidY = clampDouble(centroidY, state.halfHeight[macroIndex], state.canvasHeight - state.halfHeight[macroIndex]);

    const double oldX = state.posX[macroIndex];
    const double oldY = state.posY[macroIndex];

    const double hwx = state.halfWidth[macroIndex];
    const double hwy = state.halfHeight[macroIndex];

    state.posX[macroIndex] = centroidX;
    state.posY[macroIndex] = centroidY;
    if (!checkSingleOverlap(state, macroIndex))
    {
        const double cong = computeCongScoreForBox(
            centroidX - hwx, centroidX + hwx,
            centroidY - hwy, centroidY + hwy,
            congGrid, gridRows, gridCols,
            state.canvasWidth, state.canvasHeight);
        if (cong < threshold)
        {
            return true;
        }
    }

    const double step = std::max(state.sizeX[macroIndex], state.sizeY[macroIndex]) * 0.25;
    for (int r = 1; r <= 20; ++r)
    {
        for (int dxm = -r; dxm <= r; ++dxm)
        {
            for (int dym = -r; dym <= r; ++dym)
            {
                if (std::abs(dxm) != r && std::abs(dym) != r)
                {
                    continue;
                }
                const double trialX = clampDouble(
                    centroidX + dxm * step,
                    state.halfWidth[macroIndex],
                    state.canvasWidth - state.halfWidth[macroIndex]);
                const double trialY = clampDouble(
                    centroidY + dym * step,
                    state.halfHeight[macroIndex],
                    state.canvasHeight - state.halfHeight[macroIndex]);
                state.posX[macroIndex] = trialX;
                state.posY[macroIndex] = trialY;
                if (checkSingleOverlap(state, macroIndex))
                {
                    continue;
                }
                const double cong = computeCongScoreForBox(
                    trialX - hwx, trialX + hwx,
                    trialY - hwy, trialY + hwy,
                    congGrid, gridRows, gridCols,
                    state.canvasWidth, state.canvasHeight);
                if (cong < threshold)
                {
                    return true;
                }
            }
        }
    }

    state.posX[macroIndex] = oldX;
    state.posY[macroIndex] = oldY;
    return repairMacro(state, macroIndex);
}

py::array_t<double> swapTwoMacros(PlacerState& state, int numSwaps)
{
    std::vector<int> movableIndices;
    movableIndices.reserve(state.numHardMacros);
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        if (state.movable[i])
        {
            movableIndices.push_back(i);
        }
    }
    const int n = static_cast<int>(movableIndices.size());
    if (n >= 2)
    {
        std::uniform_int_distribution<int> idxDist(0, n - 1);
        for (int s = 0; s < numSwaps; ++s)
        {
            const int i = movableIndices[idxDist(state.rng)];
            int j = movableIndices[idxDist(state.rng)];
            if (i == j) continue;
            const double xi = state.posX[i];
            const double yi = state.posY[i];
            const double xj = state.posX[j];
            const double yj = state.posY[j];
            state.posX[i] = clampDouble(xj, state.halfWidth[i],
                state.canvasWidth - state.halfWidth[i]);
            state.posY[i] = clampDouble(yj, state.halfHeight[i],
                state.canvasHeight - state.halfHeight[i]);
            state.posX[j] = clampDouble(xi, state.halfWidth[j],
                state.canvasWidth - state.halfWidth[j]);
            state.posY[j] = clampDouble(yi, state.halfHeight[j],
                state.canvasHeight - state.halfHeight[j]);
            if (checkSingleOverlap(state, i) || checkSingleOverlap(state, j))
            {
                state.posX[i] = xi;
                state.posY[i] = yi;
                state.posX[j] = xj;
                state.posY[j] = yj;
            }
        }
    }
    py::array_t<double> result({state.numHardMacros, 2});
    auto buf = result.mutable_unchecked<2>();
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        buf(i, 0) = state.posX[i];
        buf(i, 1) = state.posY[i];
    }
    return result;
}

py::array_t<double> destroyAndRepair(PlacerState& state, int destroySize)
{
    std::vector<int> movableIndices;
    movableIndices.reserve(state.numHardMacros);
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        if (state.movable[i])
        {
            movableIndices.push_back(i);
        }
    }
    const int k = std::min<int>(destroySize, static_cast<int>(movableIndices.size()));
    if (k <= 0)
    {
        return py::array_t<double>();
    }

    std::shuffle(movableIndices.begin(), movableIndices.end(), state.rng);
    std::vector<int> destroyed(movableIndices.begin(), movableIndices.begin() + k);

    std::sort(destroyed.begin(), destroyed.end(), [&state](int a, int b) {
        return state.adjacencyIndex[a].size() > state.adjacencyIndex[b].size();
    });

    for (int idx : destroyed)
    {
        repairMacro(state, idx);
    }

    py::array_t<double> result({state.numHardMacros, 2});
    auto buf = result.mutable_unchecked<2>();
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        buf(i, 0) = state.posX[i];
        buf(i, 1) = state.posY[i];
    }
    return result;
}

py::array_t<double> destroyCongestedAndRepair(
    PlacerState& state,
    py::array_t<double> hotCellsXYW,
    int gridRows,
    int gridCols,
    int destroySize,
    py::array_t<double> congGridArr,
    int congGridRows,
    int congGridCols)
{
    std::vector<int> movableIndices;
    movableIndices.reserve(state.numHardMacros);
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        if (state.movable[i])
        {
            movableIndices.push_back(i);
        }
    }
    const int k = std::min<int>(destroySize, static_cast<int>(movableIndices.size()));
    if (k <= 0)
    {
        py::array_t<double> emptyResult({state.numHardMacros, 2});
        auto bufEmpty = emptyResult.mutable_unchecked<2>();
        for (int i = 0; i < state.numHardMacros; ++i)
        {
            bufEmpty(i, 0) = state.posX[i];
            bufEmpty(i, 1) = state.posY[i];
        }
        return emptyResult;
    }

    auto hotBuf = hotCellsXYW.unchecked<2>();
    const int numHotCells = static_cast<int>(hotBuf.shape(0));

    if (numHotCells == 0 || gridRows <= 0 || gridCols <= 0
        || state.canvasWidth <= 0.0 || state.canvasHeight <= 0.0)
    {
        return destroyAndRepair(state, destroySize);
    }

    const double cellW = state.canvasWidth / gridCols;
    const double cellH = state.canvasHeight / gridRows;

    std::vector<double> overlapScore(state.numHardMacros, 0.0);

    for (int mIdx : movableIndices)
    {
        const double mx = state.posX[mIdx];
        const double my = state.posY[mIdx];
        const double hwx = state.halfWidth[mIdx];
        const double hwy = state.halfHeight[mIdx];
        const double xMin = mx - hwx;
        const double xMax = mx + hwx;
        const double yMin = my - hwy;
        const double yMax = my + hwy;

        double score = 0.0;
        for (int h = 0; h < numHotCells; ++h)
        {
            const int row = static_cast<int>(hotBuf(h, 0));
            const int col = static_cast<int>(hotBuf(h, 1));
            const double weight = hotBuf(h, 2);
            const double cellXMin = col * cellW;
            const double cellXMax = (col + 1) * cellW;
            const double cellYMin = row * cellH;
            const double cellYMax = (row + 1) * cellH;
            const double xOverlap = std::min(cellXMax, xMax) - std::max(cellXMin, xMin);
            const double yOverlap = std::min(cellYMax, yMax) - std::max(cellYMin, yMin);
            if (xOverlap > 0.0 && yOverlap > 0.0)
            {
                score += weight * xOverlap * yOverlap;
            }
        }
        overlapScore[mIdx] = score;
    }

    std::vector<int> candidateMovable;
    candidateMovable.reserve(movableIndices.size());
    for (int idx : movableIndices)
    {
        if (overlapScore[idx] > 0.0)
        {
            candidateMovable.push_back(idx);
        }
    }

    std::vector<int> destroyed;
    destroyed.reserve(k);
    if (static_cast<int>(candidateMovable.size()) <= k)
    {
        destroyed = candidateMovable;
    }
    else
    {
        std::nth_element(
            candidateMovable.begin(),
            candidateMovable.begin() + k,
            candidateMovable.end(),
            [&overlapScore](int a, int b) {
                return overlapScore[a] > overlapScore[b];
            });
        destroyed.assign(candidateMovable.begin(), candidateMovable.begin() + k);
    }

    if (destroyed.empty())
    {
        return destroyAndRepair(state, destroySize);
    }

    std::sort(destroyed.begin(), destroyed.end(), [&state](int a, int b) {
        return state.adjacencyIndex[a].size() > state.adjacencyIndex[b].size();
    });

    const bool useAware = congGridArr.size() > 0
        && congGridRows > 0 && congGridCols > 0
        && congGridArr.ndim() == 2
        && congGridArr.shape(0) == congGridRows
        && congGridArr.shape(1) == congGridCols;

    if (useAware)
    {
        const int totalCongCells = congGridRows * congGridCols;
        std::vector<double> congFlat(totalCongCells, 0.0);
        auto congBuf = congGridArr.unchecked<2>();
        for (int row = 0; row < congGridRows; ++row)
        {
            for (int col = 0; col < congGridCols; ++col)
            {
                congFlat[row * congGridCols + col] = congBuf(row, col);
            }
        }
        std::vector<double> sortBuffer = congFlat;
        const std::size_t medianPos = sortBuffer.size() / 2;
        std::nth_element(sortBuffer.begin(), sortBuffer.begin() + medianPos, sortBuffer.end());
        const double threshold = sortBuffer[medianPos];

        for (int idx : destroyed)
        {
            repairMacroAware(state, idx, congFlat.data(), congGridRows, congGridCols, threshold);
        }
    }
    else
    {
        for (int idx : destroyed)
        {
            repairMacro(state, idx);
        }
    }

    py::array_t<double> result({state.numHardMacros, 2});
    auto buf = result.mutable_unchecked<2>();
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        buf(i, 0) = state.posX[i];
        buf(i, 1) = state.posY[i];
    }
    return result;
}

py::array_t<double> currentPositions(const PlacerState& state)
{
    py::array_t<double> result({state.numHardMacros, 2});
    auto buf = result.mutable_unchecked<2>();
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        buf(i, 0) = state.posX[i];
        buf(i, 1) = state.posY[i];
    }
    return result;
}

void setPositions(PlacerState& state, py::array_t<double> positions)
{
    auto buf = positions.unchecked<2>();
    for (int i = 0; i < state.numHardMacros; ++i)
    {
        state.posX[i] = buf(i, 0);
        state.posY[i] = buf(i, 1);
    }
}

}  // namespace

PYBIND11_MODULE(_placer_core, m)
{
    py::class_<PlacerState>(m, "PlacerState")
        .def(py::init<>())
        .def("initialize", &initializePlacerState,
             py::arg("initial_pos"), py::arg("sizes"), py::arg("movable_mask"),
             py::arg("edges"), py::arg("edge_weights"),
             py::arg("canvas_width"), py::arg("canvas_height"), py::arg("seed"))
        .def("legalize", &legalize)
        .def("legalize_min_displacement", &legalizeMinDisplacement,
             py::arg("max_iters") = 50)
        .def("sa_refine", &simulatedAnnealingRefine, py::arg("num_iters"))
        .def("sa_refine_with_stats", &simulatedAnnealingRefineWithStats,
             py::arg("num_iters"), py::arg("snapshot_every") = 0)
        .def("destroy_and_repair", &destroyAndRepair, py::arg("destroy_size"))
        .def("swap_two_macros", &swapTwoMacros, py::arg("num_swaps") = 1)
        .def("destroy_congested_and_repair", &destroyCongestedAndRepair,
             py::arg("hot_cells_xyw"), py::arg("grid_rows"), py::arg("grid_cols"),
             py::arg("destroy_size"),
             py::arg("cong_grid") = py::array_t<double>(),
             py::arg("cong_grid_rows") = 0,
             py::arg("cong_grid_cols") = 0)
        .def("current_positions", &currentPositions)
        .def("set_positions", &setPositions, py::arg("positions"));
}
