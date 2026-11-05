#include "strata/program/disk_cache_rules.hpp"

#include <cstdio>
#include <initializer_list>

using namespace strata::program::disk_cache;

namespace {
int g_fail = 0;
int g_checks = 0;
void check(bool ok, const char* what) {
    ++g_checks;
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

DiskOnlyInput one_gpu() {
    DiskOnlyInput in;
    in.serve = true;
    in.asked = true;
    in.single_stage = true;
    return in;
}

SlotSaveInput idle_slot() {
    SlotSaveInput in;
    in.cached = true;
    in.tokens = 40000;
    in.wanted = true;
    in.use_mtp = true;
    in.batch_mtp = true;
    return in;
}
}  // namespace

int main() {
    std::printf("disk_cache_rules_test:\n");
    {
        DiskOnlyInput in = one_gpu();
        check(disk_only_on(in), "one GPU, no slots: on");
        in.batch = 2;
        check(disk_only_on(in), "one GPU, --batch 2: on");
        in.batch = 8;
        check(disk_only_on(in), "one GPU, --batch 8: on");
    }
    {
        DiskOnlyInput in = one_gpu();
        in.asked = false;
        check(!disk_only_on(in), "not asked: off");
        in = one_gpu();
        in.serve = false;
        check(!disk_only_on(in), "not serving: off");
        in = one_gpu();
        in.single_stage = false;
        check(!disk_only_on(in), "layer split: off");
        in = one_gpu();
        in.multi_gpu = true;
        check(!disk_only_on(in), "several GPUs: off");
        in = one_gpu();
        in.peer_device = 1;
        check(!disk_only_on(in), "--peer-device: off");
        in.peer_device = 0;
        check(disk_only_on(in), "--peer-device 0: on");
    }
    {
        check(slot_disk_save(idle_slot()) == SlotSave::save, "idle slot with --batch-mtp: saved");
        SlotSaveInput in = idle_slot();
        in.use_mtp = false;
        in.batch_mtp = false;
        check(slot_disk_save(in) == SlotSave::save, "idle slot, no MTP: saved");
        in = idle_slot();
        in.active = true;
        check(slot_disk_save(in) == SlotSave::not_idle, "decoding slot: not saved");
        in = idle_slot();
        in.cached = false;
        check(slot_disk_save(in) == SlotSave::not_idle, "empty slot: not saved");
        in = idle_slot();
        in.partial = true;
        check(slot_disk_save(in) == SlotSave::not_idle, "a read that gave way: not saved");
        in = idle_slot();
        in.tokens = 0;
        check(slot_disk_save(in) == SlotSave::not_idle, "no tokens: not saved");
        in = idle_slot();
        in.pictures = true;
        check(slot_disk_save(in) == SlotSave::pictures, "pictures: not saved");
        in = idle_slot();
        in.wanted = false;
        check(slot_disk_save(in) == SlotSave::short_conv, "below the minimum: not saved");
        in = idle_slot();
        in.in_main = true;
        check(slot_disk_save(in) == SlotSave::held, "the main session holds it: not saved");
        in = idle_slot();
        in.on_disk = true;
        check(slot_disk_save(in) == SlotSave::held, "a disk copy holds it: not saved");
        in = idle_slot();
        in.batch_mtp = false;
        check(slot_disk_save(in) == SlotSave::no_draft, "--mtp without --batch-mtp: not saved");
        in = idle_slot();
        in.batch_df2 = true;
        check(slot_disk_save(in) == SlotSave::dflash2, "DFlash2 slot drafters: not saved");
        in = idle_slot();
        in.active = true;
        in.pictures = true;
        in.wanted = false;
        check(slot_disk_save(in) == SlotSave::not_idle, "an active slot reports not idle first");
    }
    for (SlotSave s : {SlotSave::save, SlotSave::not_idle, SlotSave::pictures, SlotSave::short_conv, SlotSave::held,
                       SlotSave::no_draft, SlotSave::dflash2})
        check(slot_disk_save_why(s)[0] != '?', "every outcome has words");
    if (g_fail != 0) {
        std::printf("disk_cache_rules_test: %d of %d checks FAILED\n", g_fail, g_checks);
        return 1;
    }
    std::printf("disk_cache_rules_test: %d checks passed\n", g_checks);
    return 0;
}
