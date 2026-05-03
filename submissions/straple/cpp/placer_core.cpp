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
        .def("sa_refine", &simulatedAnnealingRefine, py::arg("num_iters"))
        .def("destroy_and_repair", &destroyAndRepair, py::arg("destroy_size"))
        .def("current_positions", &currentPositions)
        .def("set_positions", &setPositions, py::arg("positions"));
}
