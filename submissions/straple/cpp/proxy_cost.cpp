#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

constexpr int kPinKindPort = 0;
constexpr int kPinKindHardMacroPin = 1;
constexpr int kPinKindSoftMacroPin = 2;

struct ProxyEvaluator
{
    int numHardMacros = 0;
    int numSoftMacros = 0;

    double canvasWidth = 0.0;
    double canvasHeight = 0.0;
    int gridCols = 0;
    int gridRows = 0;
    double hRoutesPerMicron = 0.0;
    double vRoutesPerMicron = 0.0;
    double hRoutingAlloc = 0.0;
    double vRoutingAlloc = 0.0;
    int smoothRange = 0;

    std::vector<double> hardWidth;
    std::vector<double> hardHeight;
    std::vector<double> softWidth;
    std::vector<double> softHeight;
    std::vector<double> softX;
    std::vector<double> softY;

    std::vector<int32_t> netPinKind;
    std::vector<int32_t> netPinOwner;
    std::vector<double> netPinOffsetX;
    std::vector<double> netPinOffsetY;
    std::vector<int32_t> netPinStart;
    std::vector<double> netWeight;
    std::vector<int32_t> netSourcePinSlot;

    std::vector<double> portPosX;
    std::vector<double> portPosY;

    std::vector<double> gridOccupied;
    std::vector<double> hRoutingCong;
    std::vector<double> vRoutingCong;
    std::vector<double> hMacroCong;
    std::vector<double> vMacroCong;

    void initialize(
        int numHardMacrosIn,
        int numSoftMacrosIn,
        double canvasWidthIn,
        double canvasHeightIn,
        int gridColsIn,
        int gridRowsIn,
        double hRoutesPerMicronIn,
        double vRoutesPerMicronIn,
        double hRoutingAllocIn,
        double vRoutingAllocIn,
        int smoothRangeIn,
        py::array_t<double> hardSizes,
        py::array_t<double> softSizes,
        py::array_t<double> softPositions,
        py::array_t<double> portPositions,
        py::array_t<int32_t> netPinKindsIn,
        py::array_t<int32_t> netPinOwnersIn,
        py::array_t<double> netPinOffsetsXIn,
        py::array_t<double> netPinOffsetsYIn,
        py::array_t<int32_t> netStartsIn,
        py::array_t<double> netWeightsIn,
        py::array_t<int32_t> netSourceSlotsIn)
    {
        numHardMacros = numHardMacrosIn;
        numSoftMacros = numSoftMacrosIn;
        canvasWidth = canvasWidthIn;
        canvasHeight = canvasHeightIn;
        gridCols = gridColsIn;
        gridRows = gridRowsIn;
        hRoutesPerMicron = hRoutesPerMicronIn;
        vRoutesPerMicron = vRoutesPerMicronIn;
        hRoutingAlloc = hRoutingAllocIn;
        vRoutingAlloc = vRoutingAllocIn;
        smoothRange = smoothRangeIn;

        auto hardSizeBuf = hardSizes.unchecked<2>();
        hardWidth.resize(numHardMacros);
        hardHeight.resize(numHardMacros);
        for (int i = 0; i < numHardMacros; ++i)
        {
            hardWidth[i] = hardSizeBuf(i, 0);
            hardHeight[i] = hardSizeBuf(i, 1);
        }

        auto softSizeBuf = softSizes.unchecked<2>();
        auto softPosBuf = softPositions.unchecked<2>();
        softWidth.resize(numSoftMacros);
        softHeight.resize(numSoftMacros);
        softX.resize(numSoftMacros);
        softY.resize(numSoftMacros);
        for (int i = 0; i < numSoftMacros; ++i)
        {
            softWidth[i] = softSizeBuf(i, 0);
            softHeight[i] = softSizeBuf(i, 1);
            softX[i] = softPosBuf(i, 0);
            softY[i] = softPosBuf(i, 1);
        }

        auto portBuf = portPositions.unchecked<2>();
        const int numPorts = static_cast<int>(portBuf.shape(0));
        portPosX.resize(numPorts);
        portPosY.resize(numPorts);
        for (int i = 0; i < numPorts; ++i)
        {
            portPosX[i] = portBuf(i, 0);
            portPosY[i] = portBuf(i, 1);
        }

        auto pinKindBuf = netPinKindsIn.unchecked<1>();
        auto pinOwnerBuf = netPinOwnersIn.unchecked<1>();
        auto offsetXBuf = netPinOffsetsXIn.unchecked<1>();
        auto offsetYBuf = netPinOffsetsYIn.unchecked<1>();
        const int numPins = static_cast<int>(pinKindBuf.shape(0));
        netPinKind.resize(numPins);
        netPinOwner.resize(numPins);
        netPinOffsetX.resize(numPins);
        netPinOffsetY.resize(numPins);
        for (int i = 0; i < numPins; ++i)
        {
            netPinKind[i] = pinKindBuf(i);
            netPinOwner[i] = pinOwnerBuf(i);
            netPinOffsetX[i] = offsetXBuf(i);
            netPinOffsetY[i] = offsetYBuf(i);
        }

        auto netStartBuf = netStartsIn.unchecked<1>();
        const int numNetStarts = static_cast<int>(netStartBuf.shape(0));
        netPinStart.resize(numNetStarts);
        for (int i = 0; i < numNetStarts; ++i)
        {
            netPinStart[i] = netStartBuf(i);
        }

        auto netWeightBuf = netWeightsIn.unchecked<1>();
        const int numNets = static_cast<int>(netWeightBuf.shape(0));
        netWeight.resize(numNets);
        for (int i = 0; i < numNets; ++i)
        {
            netWeight[i] = netWeightBuf(i);
        }

        auto sourceBuf = netSourceSlotsIn.unchecked<1>();
        netSourcePinSlot.resize(numNets);
        for (int i = 0; i < numNets; ++i)
        {
            netSourcePinSlot[i] = sourceBuf(i);
        }
    }

    inline std::pair<double, double> pinPosition(
        int kind, int owner, double offX, double offY,
        const double* hardX, const double* hardY) const
    {
        if (kind == kPinKindPort)
        {
            return {portPosX[owner], portPosY[owner]};
        }
        if (kind == kPinKindHardMacroPin)
        {
            return {hardX[owner] + offX, hardY[owner] + offY};
        }
        return {softX[owner] + offX, softY[owner] + offY};
    }

    double computeWirelength(const double* hardX, const double* hardY) const
    {
        const int numNets = static_cast<int>(netWeight.size());
        double total = 0.0;
        for (int n = 0; n < numNets; ++n)
        {
            const int begin = netPinStart[n];
            const int end = netPinStart[n + 1];
            if (end <= begin)
            {
                continue;
            }
            double minX = std::numeric_limits<double>::infinity();
            double maxX = -std::numeric_limits<double>::infinity();
            double minY = std::numeric_limits<double>::infinity();
            double maxY = -std::numeric_limits<double>::infinity();
            for (int p = begin; p < end; ++p)
            {
                auto [px, py] = pinPosition(
                    netPinKind[p], netPinOwner[p],
                    netPinOffsetX[p], netPinOffsetY[p], hardX, hardY);
                if (px < minX) minX = px;
                if (px > maxX) maxX = px;
                if (py < minY) minY = py;
                if (py > maxY) maxY = py;
            }
            total += netWeight[n] * ((maxX - minX) + (maxY - minY));
        }
        return total;
    }

    double normalizedWirelength(const double* hardX, const double* hardY) const
    {
        const double total = computeWirelength(hardX, hardY);
        const int numNets = static_cast<int>(netWeight.size());
        const int netCount = std::max(1, numNets);
        return total / ((canvasWidth + canvasHeight) * netCount);
    }

    inline std::pair<int, int> gridLocation(double xPos, double yPos) const
    {
        const double cellW = canvasWidth / gridCols;
        const double cellH = canvasHeight / gridRows;
        int row = static_cast<int>(std::floor(yPos / cellH));
        int col = static_cast<int>(std::floor(xPos / cellW));
        if (row < 0) row = 0;
        if (row >= gridRows) row = gridRows - 1;
        if (col < 0) col = 0;
        if (col >= gridCols) col = gridCols - 1;
        return {row, col};
    }

    void rasterizeMacroDensity(
        double mx, double my, double mw, double mh,
        std::vector<double>& accumulator) const
    {
        const double cellW = canvasWidth / gridCols;
        const double cellH = canvasHeight / gridRows;
        const double xMaxModule = mx + mw * 0.5;
        const double yMaxModule = my + mh * 0.5;
        const double xMinModule = mx - mw * 0.5;
        const double yMinModule = my - mh * 0.5;

        auto [urRow, urCol] = gridLocation(xMaxModule, yMaxModule);
        auto [blRow, blCol] = gridLocation(xMinModule, yMinModule);
        if (xMaxModule < 0.0 || yMaxModule < 0.0)
        {
            return;
        }
        if (xMinModule > canvasWidth || yMinModule > canvasHeight)
        {
            return;
        }
        if (blRow < 0) blRow = 0;
        if (blCol < 0) blCol = 0;
        if (urRow >= gridRows) urRow = gridRows - 1;
        if (urCol >= gridCols) urCol = gridCols - 1;

        for (int r = blRow; r <= urRow; ++r)
        {
            for (int c = blCol; c <= urCol; ++c)
            {
                const double cellXMin = c * cellW;
                const double cellXMax = (c + 1) * cellW;
                const double cellYMin = r * cellH;
                const double cellYMax = (r + 1) * cellH;
                const double xDiff = std::min(cellXMax, xMaxModule) - std::max(cellXMin, xMinModule);
                const double yDiff = std::min(cellYMax, yMaxModule) - std::max(cellYMin, yMinModule);
                if (xDiff > 0 && yDiff > 0)
                {
                    accumulator[r * gridCols + c] += xDiff * yDiff;
                }
            }
        }
    }

    double densityCost(const double* hardX, const double* hardY) const
    {
        const int totalCells = gridCols * gridRows;
        const double cellW = canvasWidth / gridCols;
        const double cellH = canvasHeight / gridRows;
        const double cellArea = cellW * cellH;

        std::vector<double> accumulator(totalCells, 0.0);

        for (int i = 0; i < numSoftMacros; ++i)
        {
            rasterizeMacroDensity(softX[i], softY[i], softWidth[i], softHeight[i], accumulator);
        }
        for (int i = 0; i < numHardMacros; ++i)
        {
            rasterizeMacroDensity(hardX[i], hardY[i], hardWidth[i], hardHeight[i], accumulator);
        }

        std::vector<double> nonZero;
        nonZero.reserve(totalCells);
        for (int i = 0; i < totalCells; ++i)
        {
            const double density = accumulator[i] / cellArea;
            if (density != 0.0)
            {
                nonZero.push_back(density);
            }
        }
        std::sort(nonZero.begin(), nonZero.end(), std::greater<double>());

        if (totalCells < 10)
        {
            if (nonZero.empty())
            {
                return 0.0;
            }
            double sum = 0.0;
            for (double d : nonZero) sum += d;
            return 0.5 * sum / nonZero.size();
        }

        const int densityCnt = static_cast<int>(std::floor(totalCells * 0.1));
        if (densityCnt <= 0)
        {
            return 0.0;
        }
        double sum = 0.0;
        const int upper = std::min<int>(densityCnt, static_cast<int>(nonZero.size()));
        for (int i = 0; i < upper; ++i) sum += nonZero[i];
        return 0.5 * sum / densityCnt;
    }

    void macroRouteOverGridCell(
        double mx, double my, double mw, double mh,
        std::vector<double>& vMacroCong,
        std::vector<double>& hMacroCong) const
    {
        const double cellW = canvasWidth / gridCols;
        const double cellH = canvasHeight / gridRows;
        const double xMaxMod = mx + mw * 0.5;
        const double yMaxMod = my + mh * 0.5;
        const double xMinMod = mx - mw * 0.5;
        const double yMinMod = my - mh * 0.5;

        auto [urRow, urCol] = gridLocation(xMaxMod, yMaxMod);
        auto [blRow, blCol] = gridLocation(xMinMod, yMinMod);

        bool partialV = false;
        bool partialH = false;

        for (int r = blRow; r <= urRow; ++r)
        {
            for (int c = blCol; c <= urCol; ++c)
            {
                const double cellXMin = c * cellW;
                const double cellXMax = (c + 1) * cellW;
                const double cellYMin = r * cellH;
                const double cellYMax = (r + 1) * cellH;
                const double xDist = std::min(cellXMax, xMaxMod) - std::max(cellXMin, xMinMod);
                const double yDist = std::min(cellYMax, yMaxMod) - std::max(cellYMin, yMinMod);
                if (xDist <= 0 || yDist <= 0)
                {
                    continue;
                }
                if (urRow != blRow)
                {
                    if ((r == blRow && std::fabs(yDist - cellH) > 1e-5) ||
                        (r == urRow && std::fabs(yDist - cellH) > 1e-5))
                    {
                        partialV = true;
                    }
                }
                if (urCol != blCol)
                {
                    if ((c == blCol && std::fabs(xDist - cellW) > 1e-5) ||
                        (c == urCol && std::fabs(xDist - cellW) > 1e-5))
                    {
                        partialH = true;
                    }
                }
                vMacroCong[r * gridCols + c] += xDist * vRoutingAlloc;
                hMacroCong[r * gridCols + c] += yDist * hRoutingAlloc;
            }
        }

        if (partialV)
        {
            const int r = urRow;
            for (int c = blCol; c <= urCol; ++c)
            {
                const double cellXMin = c * cellW;
                const double cellXMax = (c + 1) * cellW;
                const double cellYMin = r * cellH;
                const double cellYMax = (r + 1) * cellH;
                const double xDist = std::min(cellXMax, xMaxMod) - std::max(cellXMin, xMinMod);
                const double yDist = std::min(cellYMax, yMaxMod) - std::max(cellYMin, yMinMod);
                if (xDist > 0 && yDist > 0)
                {
                    vMacroCong[r * gridCols + c] -= xDist * vRoutingAlloc;
                }
            }
        }

        if (partialH)
        {
            const int c = urCol;
            for (int r = blRow; r <= urRow; ++r)
            {
                const double cellXMin = c * cellW;
                const double cellXMax = (c + 1) * cellW;
                const double cellYMin = r * cellH;
                const double cellYMax = (r + 1) * cellH;
                const double xDist = std::min(cellXMax, xMaxMod) - std::max(cellXMin, xMinMod);
                const double yDist = std::min(cellYMax, yMaxMod) - std::max(cellYMin, yMinMod);
                if (xDist > 0 && yDist > 0)
                {
                    hMacroCong[r * gridCols + c] -= yDist * hRoutingAlloc;
                }
            }
        }
    }

    static void twoPinNetRouting(
        std::pair<int, int> sourceCell,
        std::pair<int, int> sinkCell,
        double weight, int gridCols,
        std::vector<double>& vRoutingCong,
        std::vector<double>& hRoutingCong)
    {
        const int srcRow = sourceCell.first;
        const int srcCol = sourceCell.second;
        const int dstRow = sinkCell.first;
        const int dstCol = sinkCell.second;
        const int rowMin = std::min(srcRow, dstRow);
        const int rowMax = std::max(srcRow, dstRow);
        const int colMin = std::min(srcCol, dstCol);
        const int colMax = std::max(srcCol, dstCol);
        for (int colIdx = colMin; colIdx < colMax; ++colIdx)
        {
            hRoutingCong[srcRow * gridCols + colIdx] += weight;
        }
        for (int rowIdx = rowMin; rowIdx < rowMax; ++rowIdx)
        {
            vRoutingCong[rowIdx * gridCols + dstCol] += weight;
        }
    }

    static void lRouting(
        std::vector<std::pair<int, int>>& cells,
        double weight, int gridCols,
        std::vector<double>& vRoutingCong,
        std::vector<double>& hRoutingCong)
    {
        std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });
        const int y1 = cells[0].first; const int x1 = cells[0].second;
        const int y2 = cells[1].first; const int x2 = cells[1].second;
        const int y3 = cells[2].first; const int x3 = cells[2].second;
        for (int col = x1; col < x2; ++col)
        {
            hRoutingCong[y1 * gridCols + col] += weight;
        }
        for (int col = x2; col < x3; ++col)
        {
            hRoutingCong[y2 * gridCols + col] += weight;
        }
        for (int row = std::min(y1, y2); row < std::max(y1, y2); ++row)
        {
            vRoutingCong[row * gridCols + x2] += weight;
        }
        for (int row = std::min(y2, y3); row < std::max(y2, y3); ++row)
        {
            vRoutingCong[row * gridCols + x3] += weight;
        }
    }

    static void tRouting(
        std::vector<std::pair<int, int>>& cells,
        double weight, int gridCols,
        std::vector<double>& vRoutingCong,
        std::vector<double>& hRoutingCong)
    {
        std::sort(cells.begin(), cells.end());
        const int y1 = cells[0].first; const int x1 = cells[0].second;
        const int y2 = cells[1].first; const int x2 = cells[1].second;
        const int y3 = cells[2].first; const int x3 = cells[2].second;
        const int xmin = std::min({x1, x2, x3});
        const int xmax = std::max({x1, x2, x3});
        for (int col = xmin; col < xmax; ++col)
        {
            hRoutingCong[y2 * gridCols + col] += weight;
        }
        for (int row = std::min(y1, y2); row < std::max(y1, y2); ++row)
        {
            vRoutingCong[row * gridCols + x1] += weight;
        }
        for (int row = std::min(y2, y3); row < std::max(y2, y3); ++row)
        {
            vRoutingCong[row * gridCols + x3] += weight;
        }
    }

    void threePinNetRouting(
        std::vector<std::pair<int, int>>& cells, double weight,
        std::vector<double>& vRoutingCong,
        std::vector<double>& hRoutingCong) const
    {
        std::vector<std::pair<int, int>> sorted = cells;
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });
        const int y1 = sorted[0].first; const int x1 = sorted[0].second;
        const int y2 = sorted[1].first; const int x2 = sorted[1].second;
        const int y3 = sorted[2].first; const int x3 = sorted[2].second;

        if (x1 < x2 && x2 < x3 && std::min(y1, y3) < y2 && std::max(y1, y3) > y2)
        {
            lRouting(sorted, weight, gridCols, vRoutingCong, hRoutingCong);
        }
        else if (x2 == x3 && x1 < x2 && y1 < std::min(y2, y3))
        {
            for (int col = x1; col < x2; ++col)
            {
                hRoutingCong[y1 * gridCols + col] += weight;
            }
            for (int row = y1; row < std::max(y2, y3); ++row)
            {
                vRoutingCong[row * gridCols + x2] += weight;
            }
        }
        else if (y2 == y3)
        {
            for (int col = x1; col < x2; ++col)
            {
                hRoutingCong[y1 * gridCols + col] += weight;
            }
            for (int col = x2; col < x3; ++col)
            {
                hRoutingCong[y2 * gridCols + col] += weight;
            }
            for (int row = std::min(y2, y1); row < std::max(y2, y1); ++row)
            {
                vRoutingCong[row * gridCols + x2] += weight;
            }
        }
        else
        {
            tRouting(sorted, weight, gridCols, vRoutingCong, hRoutingCong);
        }
    }

    void smoothRoutingCong(
        std::vector<double>& vRoutingCong,
        std::vector<double>& hRoutingCong) const
    {
        const int totalCells = gridCols * gridRows;
        std::vector<double> tempV(totalCells, 0.0);
        std::vector<double> tempH(totalCells, 0.0);

        for (int row = 0; row < gridRows; ++row)
        {
            for (int col = 0; col < gridCols; ++col)
            {
                int lp = col - smoothRange;
                if (lp < 0) lp = 0;
                int rp = col + smoothRange;
                if (rp >= gridCols) rp = gridCols - 1;
                const int gcellCnt = rp - lp + 1;
                const double val = vRoutingCong[row * gridCols + col] / gcellCnt;
                for (int ptr = lp; ptr <= rp; ++ptr)
                {
                    tempV[row * gridCols + ptr] += val;
                }
            }
        }
        vRoutingCong = tempV;

        for (int row = 0; row < gridRows; ++row)
        {
            for (int col = 0; col < gridCols; ++col)
            {
                int lp = row - smoothRange;
                if (lp < 0) lp = 0;
                int up = row + smoothRange;
                if (up >= gridRows) up = gridRows - 1;
                const int gcellCnt = up - lp + 1;
                const double val = hRoutingCong[row * gridCols + col] / gcellCnt;
                for (int ptr = lp; ptr <= up; ++ptr)
                {
                    tempH[ptr * gridCols + col] += val;
                }
            }
        }
        hRoutingCong = tempH;
    }

    double congestionCost(const double* hardX, const double* hardY) const
    {
        const int totalCells = gridCols * gridRows;
        const double cellW = canvasWidth / gridCols;
        const double cellH = canvasHeight / gridRows;
        const double gridVRoutes = cellW * vRoutesPerMicron;
        const double gridHRoutes = cellH * hRoutesPerMicron;

        std::vector<double> vRoutingCong(totalCells, 0.0);
        std::vector<double> hRoutingCong(totalCells, 0.0);
        std::vector<double> vMacroCong(totalCells, 0.0);
        std::vector<double> hMacroCong(totalCells, 0.0);

        for (int i = 0; i < numHardMacros; ++i)
        {
            macroRouteOverGridCell(
                hardX[i], hardY[i], hardWidth[i], hardHeight[i],
                vMacroCong, hMacroCong);
        }

        const int numNets = static_cast<int>(netWeight.size());
        for (int n = 0; n < numNets; ++n)
        {
            const int begin = netPinStart[n];
            const int end = netPinStart[n + 1];
            if (end <= begin)
            {
                continue;
            }
            const int sourcePinSlot = netSourcePinSlot[n];
            const int sourcePinIdx = begin + sourcePinSlot;
            auto [sx, sy] = pinPosition(
                netPinKind[sourcePinIdx], netPinOwner[sourcePinIdx],
                netPinOffsetX[sourcePinIdx], netPinOffsetY[sourcePinIdx], hardX, hardY);
            auto sourceCell = gridLocation(sx, sy);

            std::vector<std::pair<int, int>> nodeCells;
            nodeCells.reserve(end - begin);
            nodeCells.push_back(sourceCell);
            for (int p = begin; p < end; ++p)
            {
                if (p == sourcePinIdx) continue;
                auto [px, py] = pinPosition(
                    netPinKind[p], netPinOwner[p],
                    netPinOffsetX[p], netPinOffsetY[p], hardX, hardY);
                auto cell = gridLocation(px, py);
                if (std::find(nodeCells.begin(), nodeCells.end(), cell) == nodeCells.end())
                {
                    nodeCells.push_back(cell);
                }
            }

            const double weight = netWeight[n];
            if (nodeCells.size() == 2)
            {
                twoPinNetRouting(
                    nodeCells[0], nodeCells[1],
                    weight, gridCols, vRoutingCong, hRoutingCong);
            }
            else if (nodeCells.size() == 3)
            {
                threePinNetRouting(nodeCells, weight, vRoutingCong, hRoutingCong);
            }
            else if (nodeCells.size() > 3)
            {
                for (const auto& c : nodeCells)
                {
                    if (c == sourceCell) continue;
                    twoPinNetRouting(
                        sourceCell, c, weight, gridCols,
                        vRoutingCong, hRoutingCong);
                }
            }
        }

        for (int i = 0; i < totalCells; ++i)
        {
            vRoutingCong[i] /= gridVRoutes;
            hRoutingCong[i] /= gridHRoutes;
            vMacroCong[i] /= gridVRoutes;
            hMacroCong[i] /= gridHRoutes;
        }

        smoothRoutingCong(vRoutingCong, hRoutingCong);

        for (int i = 0; i < totalCells; ++i)
        {
            vRoutingCong[i] += vMacroCong[i];
            hRoutingCong[i] += hMacroCong[i];
        }

        std::vector<double> combined;
        combined.reserve(2 * totalCells);
        combined.insert(combined.end(), vRoutingCong.begin(), vRoutingCong.end());
        combined.insert(combined.end(), hRoutingCong.begin(), hRoutingCong.end());
        std::sort(combined.begin(), combined.end(), std::greater<double>());

        const int cnt = static_cast<int>(std::floor(combined.size() * 0.05));
        if (cnt == 0)
        {
            return combined.empty() ? 0.0 : combined.front();
        }
        double sum = 0.0;
        for (int i = 0; i < cnt; ++i) sum += combined[i];
        return sum / cnt;
    }

    double evaluate(py::array_t<double> hardPositions) const
    {
        auto buf = hardPositions.unchecked<2>();
        std::vector<double> hardX(numHardMacros);
        std::vector<double> hardY(numHardMacros);
        for (int i = 0; i < numHardMacros; ++i)
        {
            hardX[i] = buf(i, 0);
            hardY[i] = buf(i, 1);
        }
        const double wl = normalizedWirelength(hardX.data(), hardY.data());
        const double dens = densityCost(hardX.data(), hardY.data());
        const double cong = congestionCost(hardX.data(), hardY.data());
        return wl + dens + cong;
    }

    py::tuple evaluateBreakdown(py::array_t<double> hardPositions) const
    {
        auto buf = hardPositions.unchecked<2>();
        std::vector<double> hardX(numHardMacros);
        std::vector<double> hardY(numHardMacros);
        for (int i = 0; i < numHardMacros; ++i)
        {
            hardX[i] = buf(i, 0);
            hardY[i] = buf(i, 1);
        }
        const double wl = normalizedWirelength(hardX.data(), hardY.data());
        const double dens = densityCost(hardX.data(), hardY.data());
        const double cong = congestionCost(hardX.data(), hardY.data());
        return py::make_tuple(wl + dens + cong, wl, dens, cong);
    }
};

}  // namespace

PYBIND11_MODULE(_proxy_cost, m)
{
    py::class_<ProxyEvaluator>(m, "ProxyEvaluator")
        .def(py::init<>())
        .def("initialize", &ProxyEvaluator::initialize,
             py::arg("num_hard"), py::arg("num_soft"),
             py::arg("canvas_width"), py::arg("canvas_height"),
             py::arg("grid_cols"), py::arg("grid_rows"),
             py::arg("h_routes_per_micron"), py::arg("v_routes_per_micron"),
             py::arg("h_routing_alloc"), py::arg("v_routing_alloc"),
             py::arg("smooth_range"),
             py::arg("hard_sizes"), py::arg("soft_sizes"),
             py::arg("soft_positions"), py::arg("port_positions"),
             py::arg("net_pin_kinds"), py::arg("net_pin_owners"),
             py::arg("net_pin_offsets_x"), py::arg("net_pin_offsets_y"),
             py::arg("net_starts"), py::arg("net_weights"),
             py::arg("net_source_slots"))
        .def("evaluate", &ProxyEvaluator::evaluate, py::arg("hard_positions"))
        .def("evaluate_breakdown", &ProxyEvaluator::evaluateBreakdown, py::arg("hard_positions"));
}
