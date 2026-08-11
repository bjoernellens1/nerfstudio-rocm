Smoke test logs land in `logs/` (gitignored) when `scripts/smoke-test.sh` runs.
See `ROCM.md` §Phase 1 for the acceptance criteria these logs are checked against.

`splatfacto.log` — Splatfacto training on the bonsai mip-nerf360 scene
(7000 iterations, downscale-factor 4 with round rounding mode from original
images/ folder). See `ROCM.md` for the gsplat verification this depends on.

`tcnn_wave32_correctness.py` — standalone tiny-rocm-nn (tinycudann) wave32/RDNA
numerical-correctness check, independent of `smoke-test.sh`. Run directly
against a real GPU; writes `logs/tcnn-wave32-correctness.log`. The companion
Instant-NGP end-to-end run (Task 4) writes `logs/tcnn-instant-ngp.log`. See
`ROCM.md`'s "tiny-rocm-nn: what's actually there" section for the results.
