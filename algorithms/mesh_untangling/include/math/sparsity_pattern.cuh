#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/base.cuh"

namespace mu::energy
{

template <typename Index>
class HessianSparsityPattern
{
public:
    using IndexType = Index;

    auto Reset(std::size_t rowCount) -> void
    {
        rows_.assign(rowCount, {});
        rowOffsets_.clear();
        columnIndices_.clear();
        finalized_ = false;
    }

    MU_HOST_DEV_FUNC auto RowCount() const -> std::size_t
    {
        if (finalized_)
        {
            return rowOffsets_.empty() ? 0 : rowOffsets_.size() - 1;
        }
        return rows_.size();
    }

    auto Insert(Index row, Index col) -> void
    {
        if (static_cast<std::size_t>(row) >= rows_.size())
        {
            return;
        }
        auto& columns = rows_[row];
        const auto it = std::lower_bound(columns.begin(), columns.end(), col);
        if (it == columns.end() || *it != col)
        {
            columns.insert(it, col);
            finalized_ = false;
        }
    }

    auto Finalize() -> void
    {
        rowOffsets_.assign(rows_.size() + 1, Index{0});
        std::size_t running = 0;
        for (std::size_t r = 0; r < rows_.size(); ++r)
        {
            running += rows_[r].size();
            rowOffsets_[r + 1] = static_cast<Index>(running);
        }

        columnIndices_.clear();
        columnIndices_.reserve(running);
        for (const auto& cols : rows_)
        {
            columnIndices_.insert(columnIndices_.end(), cols.begin(), cols.end());
        }

        finalized_ = true;
    }

    MU_HOST_DEV_FUNC auto Finalized() const -> bool
    {
        return finalized_;
    }

    MU_HOST_DEV_FUNC auto NonZeroCount() const -> std::size_t
    {
        return columnIndices_.size();
    }

    MU_HOST_DEV_FUNC auto RowOffsets() const -> const std::vector<Index>&
    {
        return rowOffsets_;
    }

    MU_HOST_DEV_FUNC auto ColumnIndices() const -> const std::vector<Index>&
    {
        return columnIndices_;
    }

    auto AssignFinalized(std::vector<Index> rowOffsets, std::vector<Index> columnIndices) -> void
    {
        rows_.clear();
        rowOffsets_ = std::move(rowOffsets);
        columnIndices_ = std::move(columnIndices);
        finalized_ = true;
    }

private:
    std::vector<std::vector<Index>> rows_;
    std::vector<Index> rowOffsets_;
    std::vector<Index> columnIndices_;
    bool finalized_ = false;
};

template <typename Index>
class HessianRowIndexer
{
public:
    HessianRowIndexer(const std::vector<Index>& rowOffsets, const std::vector<Index>& columnIndices)
        : rowOffsets_(rowOffsets), columnIndices_(columnIndices)
    {
    }

    auto Slot(Index row, Index col) const -> std::optional<Index>
    {
        if (static_cast<std::size_t>(row + 1) >= rowOffsets_.size())
        {
            return std::nullopt;
        }
        const auto begin = columnIndices_.begin() + static_cast<std::ptrdiff_t>(rowOffsets_[row]);
        const auto end = columnIndices_.begin() + static_cast<std::ptrdiff_t>(rowOffsets_[row + 1]);
        const auto it = std::lower_bound(begin, end, col);
        if (it == end || *it != col)
        {
            return std::nullopt;
        }
        return static_cast<Index>(std::distance(columnIndices_.begin(), it));
    }

private:
    const std::vector<Index>& rowOffsets_;
    const std::vector<Index>& columnIndices_;
};

template <typename Index>
struct HessianColumnBuffer
{
    std::string tag;
    std::vector<Index> hostIndices;
    Index* devicePtr = nullptr;
};

template <typename Index>
struct HessianAssemblyContext
{
    HessianAssemblyContext(const std::vector<Index>& rowOffsets,
                           const std::vector<Index>& columnIndices)
        : rowOffsets(rowOffsets),
          columnIndices(columnIndices),
          rowIndexer(rowOffsets, columnIndices)
    {
    }

    auto AddBuffer(const std::string& tag, std::size_t count) -> Index*
    {
        columnBuffers.emplace_back();
        auto& buffer = columnBuffers.back();
        buffer.tag = tag;
        buffer.hostIndices.resize(count);
        return buffer.hostIndices.data();
    }

    const std::vector<Index>& rowOffsets;
    const std::vector<Index>& columnIndices;
    HessianRowIndexer<Index> rowIndexer;
    std::vector<HessianColumnBuffer<Index>> columnBuffers;
};

template <typename Index>
struct HessianColumnBufferView
{
    std::string tag;
    const Index* devicePtr = nullptr;
};

template <typename Index, typename Scalar>
struct HessianAssemblyView
{
    const Index* rowOffsetsDevice = nullptr;
    const Index* columnIndicesDevice = nullptr;
    Scalar* valuesDevice = nullptr;
    std::vector<HessianColumnBufferView<Index>> extraColumns;

    auto GetColumnDevice(const std::string& tag) const -> const Index*
    {
        const auto it = std::find_if(extraColumns.begin(), extraColumns.end(),
                                     [&](const auto& entry)
                                     {
                                         return entry.tag == tag;
                                     });
        return it == extraColumns.end() ? nullptr : it->devicePtr;
    }
};

template <typename Pattern>
struct HessianSparsityUpdateTarget
{
    Pattern* staticPattern = nullptr;
    Pattern* dynamicPattern = nullptr;
};

}
