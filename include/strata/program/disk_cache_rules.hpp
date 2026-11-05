#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::program::disk_cache {

struct DiskOnlyInput {
    bool serve = false;
    bool asked = false;
    bool single_stage = false;
    bool multi_gpu = false;
    int64_t batch = 0;
    int peer_device = -1;
};

inline bool disk_only_on(const DiskOnlyInput& in) {
    return in.serve && in.asked && in.single_stage && !in.multi_gpu && in.peer_device < 1;
}

enum class SlotSave {
    save,
    not_idle,
    pictures,
    short_conv,
    held,
    no_draft,
    dflash2,
};

struct SlotSaveInput {
    bool active = false;
    bool cached = false;
    bool partial = false;
    bool pictures = false;
    size_t tokens = 0;
    bool wanted = false;
    bool in_main = false;
    bool on_disk = false;
    bool use_mtp = false;
    bool batch_mtp = false;
    bool batch_df2 = false;
};

inline SlotSave slot_disk_save(const SlotSaveInput& in) {
    if (in.active || !in.cached || in.partial || in.tokens == 0) return SlotSave::not_idle;
    if (in.pictures) return SlotSave::pictures;
    if (!in.wanted) return SlotSave::short_conv;
    if (in.in_main || in.on_disk) return SlotSave::held;
    if (in.batch_df2) return SlotSave::dflash2;
    if (in.use_mtp && !in.batch_mtp) return SlotSave::no_draft;
    return SlotSave::save;
}

inline const char* slot_disk_save_why(SlotSave s) {
    switch (s) {
        case SlotSave::save: return "saved";
        case SlotSave::not_idle: return "not idle";
        case SlotSave::pictures: return "it has pictures";
        case SlotSave::short_conv: return "below the minimum";
        case SlotSave::held: return "already held";
        case SlotSave::no_draft: return "no per-slot draft K/V without --batch-mtp";
        case SlotSave::dflash2: return "DFlash2 slot drafters";
    }
    return "?";
}

}  // namespace strata::program::disk_cache
