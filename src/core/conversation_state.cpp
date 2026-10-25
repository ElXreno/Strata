#include "strata/core/conversation_snapshot.hpp"
#include "conversation_checked.hpp"
#include "conversation_copy.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot synchronize: ") + cudaGetErrorString(status);
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error, bool queued = false) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    const auto status = conversation_detail::copy_nonblocking(dst, src, bytes, queued);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

bool state_sizes_valid(size_t tokens, const std::vector<ConversationImageKey>& imgs,
                       const std::array<uint64_t, 5>& sizes, const SessionState& ss, const ModelGeometry& g,
                       std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, tokens, z, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (sizes[0] != z.gdn || sizes[1] != (ss.ple_hist ? z.ple : 0) ||
        sizes[2] != layers * z.tail || sizes[3] != layers * z.dead ||
        sizes[4] != layers * z.block_pos || !image_keys(imgs, tokens))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool live_validate(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                   const SessionState& ss, const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, ids.size(), z, error)) return false;
    if (ids.empty() || !image_keys(images, ids.size()) ||
        std::any_of(ids.begin(), ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    return true;
}

bool view_checkpoint(const ConversationCheckpoint& c, const std::vector<int32_t>& ids,
                     const std::vector<ConversationImageKey>& images, const SessionState& ss,
                     const ModelGeometry& g, std::string& error) {
    if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size() > ids.size() ||
        !std::equal(c.ids.begin(), c.ids.end(), ids.begin()))
        return fail(error, "checkpoint is not a live token prefix");
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    size_t image = 0;
    for (const auto& key : images) {
        if ((uint64_t) key.start >= c.ids.size()) break;
        if (image >= c.imgs.size() || !(c.imgs[image++] == key))
            return fail(error, "checkpoint image identity differs");
    }
    if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    return true;
}

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    if (!live_validate(view.ids, view.images, ss, g, error)) return false;
    for (const auto& c : view.checkpoints)
        if (!view_checkpoint(c, view.ids, view.images, ss, g, error)) return false;
    return true;
}

bool stream_view_validate(const ConversationStreamView& view, const SessionState& ss,
                          const ModelGeometry& g, std::string& error) {
    if (!live_validate(view.ids, view.images, ss, g, error)) return false;
    for (const ConversationCheckpoint* c : view.checkpoints) {
        if (!c) return fail(error, "missing checkpoint");
        if (!view_checkpoint(*c, view.ids, view.images, ss, g, error)) return false;
    }
    return true;
}

bool state_source(SessionStateSource& out, const SessionState& ss, const ModelGeometry& g,
                  const std::vector<int32_t>& ids, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, ids.size(), z, error)) return false;
#if defined(STRATA_GPU_METAL)
    auto held = std::make_shared<ConversationCheckpoint>();   // the decode window's state, captured whole
    held->ids = ids;
    if (!conversation_checkpoint_save(*held, ss, g, error)) return false;
    SessionStateSource s = session_state_source(*held);
    s.read = [held, read = std::move(s.read)](size_t part, size_t offset, void* dst, size_t n) {
        return read(part, offset, dst, n);
    };
    out = std::move(s);
    return true;
#endif
    const size_t layers = owned_qsa(ss);
    std::array<std::vector<const uint8_t*>, 3> rows;   // tails, dead, block_pos per owned QSA layer
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        rows[0].push_back(reinterpret_cast<const uint8_t*>(st.idx_tail));
        rows[1].push_back(reinterpret_cast<const uint8_t*>(st.idx_dead));
        rows[2].push_back(reinterpret_cast<const uint8_t*>(st.idx_block_pos));
    }
    const std::array<const uint8_t*, 2> whole = {reinterpret_cast<const uint8_t*>(ss.gdn_state),
                                                 reinterpret_cast<const uint8_t*>(ss.ple_hist)};
    const std::array<size_t, 3> row = {z.tail, z.dead, z.block_pos};
    SessionStateSource s;
    s.sizes = {z.gdn, ss.ple_hist ? z.ple : 0, layers * z.tail, layers * z.dead, layers * z.block_pos};
    const auto sizes = s.sizes;
    auto why = std::make_shared<std::string>();
    s.read = [rows, whole, row, sizes, why](size_t part, size_t offset, void* dst, size_t n) {
        if (part >= 5 || offset > sizes[part] || n > sizes[part] - offset) { *why = "running-state read out of range"; return false; }
        uint8_t* d = static_cast<uint8_t*>(dst);
        if (part < 2) return copy(d, whole[part] + offset, n, *why);
        while (n) {
            const size_t r = row[part - 2], j = offset / r, within = offset % r, c = std::min(n, r - within);
            if (!copy(d, rows[part - 2][j] + within, c, *why)) return false;
            d += c; offset += c; n -= c;
        }
        return true;
    };
    s.error = [why] { return *why; };
    out = std::move(s);
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    return state_sizes_valid(c.ids.size(), c.imgs,
                             {c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()}, ss, g,
                             error);
}

#if defined(STRATA_GPU_METAL)
}  // namespace strata::core
namespace strata::metal_engine {
// the running state lives in the Metal decode window (metal/src/engine/window.cpp), not in the session's buffers
bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& error);
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& error);
}  // namespace strata::metal_engine
namespace strata::core {
#endif

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) ss; (void) g;
    return strata::metal_engine::window_checkpoint_save(c, error);
#endif
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error) ||
        !copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
            !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
            !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) g;
    if (!strata::metal_engine::window_checkpoint_restore(c, error)) return false;
    ss.ple_prev[0] = c.ids.size() >= 2 ? c.ids[c.ids.size() - 2] : -1;
    ss.ple_prev[1] = c.ids.size() >= 1 ? c.ids[c.ids.size() - 1] : -1;
    return true;
#endif
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!sync(error)) return false;   // the copies below no longer ride the legacy stream's implicit ordering
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

bool conversation_checkpoint_copy(const SessionState& from, SessionState& to, const std::vector<int32_t>& ids,
                                  const ModelGeometry& g, std::string& error, bool queued) {
    // conversation_checkpoint_save(from) + conversation_checkpoint_restore(to) without the host image, for two
    // sessions carved alike on the same device (a stage's own session and its batch slots): the same bytes into
    // the same places, through copy() like the restore (the private non-blocking stream, complete on return unless
    // `queued`: then the caller synchronizes the device).
    ConversationStateSizes zf, zt;
    if (!checkpoint_targets(from, g, ids.size(), zf, error) || !checkpoint_targets(to, g, ids.size(), zt, error))
        return false;
    if (zf.gdn != zt.gdn || zf.tail != zt.tail || zf.dead != zt.dead || zf.block_pos != zt.block_pos ||
        owned_qsa(from) != owned_qsa(to) || (from.ple_hist != nullptr) != (to.ple_hist != nullptr))
        return fail(error, "direct copy between differently carved sessions");
    auto acopy = [&](void* dst, const void* src, size_t bytes) { return copy(dst, src, bytes, error, queued); };
    if (!acopy(to.gdn_state, from.gdn_state, zf.gdn) ||
        !acopy(to.ple_hist, from.ple_hist, from.ple_hist ? zf.ple : 0)) return false;
    for (size_t j = 0; j < owned_qsa(from); ++j) {
        const auto& a = owned(from, j);
        const auto& b = owned(to, j);
        if (!acopy(b.idx_tail, a.idx_tail, zf.tail) || !acopy(b.idx_dead, a.idx_dead, zf.dead) ||
            !acopy(b.idx_block_pos, a.idx_block_pos, zf.block_pos)) return false;
        if (!ids.empty()) {   // restore writes the saved `dead` row into the pooled row the next token fills
            const size_t row = ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!acopy(b.idx_pooled + row * g.idx_key_dim, a.idx_dead, zf.dead)) return false;
        }
    }
    const size_t tokens = ids.size();
    to.ple_prev[0] = tokens >= 2 ? ids[tokens - 2] : -1;
    to.ple_prev[1] = tokens >= 1 ? ids[tokens - 1] : -1;
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState* draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!view_validate(view, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + (draft ? 1 : 0), sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i < qsa + (draft ? 1 : 0); ++i) {
        const auto& st = i == qsa ? *draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState* draft, size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss) + (draft ? 1 : 0);
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i < owned_qsa(ss);
        const auto& st = index ? owned(ss, i) : *draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss);
    captured.kv.resize(layers + (draft ? 1 : 0));
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (draft && !conversation_kv_save(captured.kv.back(), *draft, g, upto, false, error,
                                       std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                   const QsaState& draft, std::string& error) {
    if (!view_validate(view, ss, g, error) || !sync(error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss);
    std::vector<SessionKvSource> out(layers + 1);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_source(out.back(), draft, g, upto, false, error)) return false;
    meta = std::move(captured);
    sources = std::move(out);
    return true;
}

bool conversation_snapshot_stream(SessionSaveSources& save, const ConversationStreamView& view,
                                  const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                  std::string& error) {
    if (!stream_view_validate(view, ss, g, error) || !sync(error)) return false;
    SessionSaveSources out;
    out.meta.geometry = geometry_key(g);
    out.meta.layer_lo = ss.layer_lo; out.meta.layer_hi = ss.layer_hi;
    out.meta.live.ids = view.ids; out.meta.live.imgs = view.images;
    out.meta.cvec = view.cvec;
    out.checkpoints = view.checkpoints;
    if (!state_source(out.state, ss, g, view.ids, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss);
    out.kv.resize(layers + 1);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out.kv[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_source(out.kv.back(), draft, g, upto, false, error)) return false;
    save = std::move(out);
    return true;
}

bool conversation_state_target(const SavedConversation& image, const std::array<uint64_t, 5>& sizes,
                               SessionState& ss, const ModelGeometry& g, ConversationStateTarget& target,
                               std::string& error) {
    if (!state_sizes_valid(image.live.ids.size(), image.live.imgs, sizes, ss, g, error)) return false;
    ConversationStateTarget t;
    t.session = &ss; t.geometry = &g; t.sizes = sizes; t.tokens = (int64_t) image.live.ids.size();
#if defined(STRATA_GPU_METAL)
    t.staged = std::make_shared<ConversationCheckpoint>();
    const std::array<std::vector<uint8_t>*, 5> parts = {&t.staged->gdn, &t.staged->ple, &t.staged->tails,
                                                        &t.staged->dead, &t.staged->block_pos};
    for (size_t i = 0; i < parts.size(); ++i) parts[i]->resize((size_t) sizes[i]);
#endif
    target = std::move(t);
    return true;
}

bool ConversationStateTarget::apply(size_t part, uint64_t offset, const void* data, size_t n,
                                    std::string& error) const {
    if (!session || !geometry || part >= 5 || offset > sizes[part] || n > sizes[part] - offset)
        return fail(error, "running-state block outside its part");
    if (!n) return true;
    const uint8_t* src = static_cast<const uint8_t*>(data);
    if (staged) {
        const std::array<std::vector<uint8_t>*, 5> parts = {&staged->gdn, &staged->ple, &staged->tails,
                                                            &staged->dead, &staged->block_pos};
        if (parts[part]->size() != sizes[part]) return fail(error, "running-state staging size differs");
        std::memcpy(parts[part]->data() + offset, src, n);
        return true;
    }
    ConversationStateSizes z;
    if (!conversation_session_sizes(*geometry, *session, z, error)) return false;
    if (part == 0) return copy(reinterpret_cast<uint8_t*>(session->gdn_state) + offset, src, n, error);
    if (part == 1) return copy(reinterpret_cast<uint8_t*>(session->ple_hist) + offset, src, n, error);
    const size_t r = part == 2 ? z.tail : part == 3 ? z.dead : z.block_pos;
    const size_t pooled_row = (size_t) tokens / strata::kernels::qsa_real_shapes().idx_block;
    while (n) {
        const size_t j = (size_t) (offset / r), within = (size_t) (offset % r), c = std::min(n, r - within);
        const auto& st = owned(*session, j);
        uint8_t* dst = part == 2 ? reinterpret_cast<uint8_t*>(st.idx_tail)
                     : part == 3 ? reinterpret_cast<uint8_t*>(st.idx_dead)
                                 : reinterpret_cast<uint8_t*>(st.idx_block_pos);
        if (!copy(dst + within, src, c, error)) return false;
        // as conversation_checkpoint_restore: the saved `dead` row is the pooled row the next token fills
        if (part == 3 && tokens > 0 &&
            !copy(reinterpret_cast<uint8_t*>(st.idx_pooled + pooled_row * geometry->idx_key_dim) + within, src, c,
                  error)) return false;
        src += c; offset += c; n -= c;
    }
    return true;
}

bool ConversationStateTarget::finish(const std::vector<int32_t>& ids, std::string& error) const {
    if (!session || !geometry || (int64_t) ids.size() != tokens) return fail(error, "unbound running-state target");
    if (staged) {
        staged->ids = ids;
        return conversation_checkpoint_restore(*staged, *session, *geometry, error);
    }
    const size_t n = ids.size();
    session->ple_prev[0] = n >= 2 ? ids[n - 2] : -1;
    session->ple_prev[1] = n >= 1 ? ids[n - 1] : -1;
    return sync(error);
}

bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& ss, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0) return fail(error, "invalid session cells");
    const uint64_t tokens = std::min<uint64_t>(max_tokens, (uint64_t) ss.max_cells);
    const size_t layers = owned_qsa(ss);
    if ((owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    size_t tails = 0, dead = 0, block_pos = 0;
    if (!product(tails, {layers, z.tail}) || !product(dead, {layers, z.dead}) ||
        !product(block_pos, {layers, z.block_pos})) return fail(error, "indexer byte count overflow");
    SessionReadLimits l = limits;   // keeps the caller's admit / progress / max_file_bytes
    l.max_tokens = tokens;
    l.max_checkpoints = max_checkpoints;
    l.max_kv_layers = layers + 1;
    l.geometry = geometry_key(g);
    l.layer_range = std::make_pair(ss.layer_lo, ss.layer_hi);
    l.max_state_bytes = {z.gdn, ss.ple_hist ? z.ple : 0, tails, dead, block_pos};
    l.max_kv_bytes.assign(layers + 1, {});
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        const int64_t upto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(st.max_cells, 0));
        if (!conversation_kv_part_sizes(st, g, upto, true, l.max_kv_bytes[j], error)) return false;
    }
    const int64_t dupto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(draft.max_cells, 0));
    if (!conversation_kv_part_sizes(draft, g, dupto, false, l.max_kv_bytes.back(), error)) return false;
    limits = std::move(l);
    return true;
}

namespace {
bool meta_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                   const QsaState* draft, const std::array<uint64_t, 5>* live_state, std::string& error) {
    if (!image.live.stage_parts.empty())
        return fail(error, "a running checkpoint's stage parts belong in the stage images, not in live.stage_parts");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error)) return false;
    if (live_state) {
        const auto& l = image.live;
        if (!l.gdn.empty() || !l.ple.empty() || !l.tails.empty() || !l.dead.empty() || !l.block_pos.empty())
            return fail(error, "the live running state is both in the image and on disk");
        if (!state_sizes_valid(l.ids.size(), l.imgs, *live_state, ss, g, error)) return false;
    } else if (!conversation_checkpoint_validate(image.live, ss, g, error)) {
        return false;
    }
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + (draft ? 1 : 0)) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate_meta(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return !draft || conversation_kv_validate_meta(image.kv.back(), *draft, g, upto, false, error);
}
} // namespace

bool conversation_snapshot_validate_meta(const SavedConversation& image, const SessionState& ss,
                                         const ModelGeometry& g, const QsaState* draft, std::string& error) {
    return meta_validate(image, ss, g, draft, nullptr, error);
}

bool conversation_snapshot_validate_stream(const SavedConversation& image, const std::array<uint64_t, 5>& live_state,
                                           const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                           std::string& error) {
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return meta_validate(image, ss, g, &draft, &live_state, error);
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error) {
    if (!conversation_snapshot_validate_meta(image, ss, g, draft, error)) return false;
    const size_t layers = owned_qsa(ss);
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return !draft || conversation_kv_validate(image.kv.back(), *draft, g, upto, false, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error) {
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if ((draft && !conversation_kv_restore(image.kv.back(), *draft, g, upto, false, error)) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}

// the draft layer's K/V included (the stage that owns the draft head, or no layer split)
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_bytes(view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error) {
    return conversation_snapshot_capture_bytes(reuse, view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& ss,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    return conversation_snapshot_save(image, view, ss, g, &draft, error, std::move(reuse), reused_bytes);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error) {
    // the whole-session form: a layer split's image (stage images) is restored stage by stage, never through here
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate(image, ss, g, &draft, error);
}
bool conversation_snapshot_validate_meta(const SavedConversation& image, const SessionState& ss,
                                         const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate_meta(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore_prefix(image, upto, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore_prefix(const SavedConversation& image, int64_t upto,
                                                          SessionState& ss, const ModelGeometry& g,
                                                          const QsaState* draft, std::string& error) {
    // a layer split's image (stage images) is restored stage by stage, never in part
    if (!image.stage_images.empty()) {
        fail(error, "a layer split's image restored as a single session");
        return ConversationRestore::invalid;
    }
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    const int64_t captured = (int64_t) image.live.ids.size();
    if (upto < 1 || upto > captured) {
        fail(error, "the borrowed prefix is not inside the parked conversation");
        return ConversationRestore::invalid;
    }
    if (!sync(error)) return ConversationRestore::transfer_failed;
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore_prefix(image.kv[j], owned(ss, j), g, captured, upto, true, error))
            return ConversationRestore::transfer_failed;
    if (draft && !conversation_kv_restore_prefix(image.kv.back(), *draft, g, captured, upto, false, error))
        return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
